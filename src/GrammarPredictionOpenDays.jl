module GrammarPredictionOpenDays

import HTTP, JSON, CSV
import Base: +, *, zero, one

using DataFrames, LogProbs
using Distributions: Categorical
using Statistics: mean
using DataStructures: counter
using ProgressMeter: @showprogress
using Underscores: @_
using Memoize: @memoize

export run_main

#############
### Trees ###
#############

abstract type Tree{L} end
struct Binary{L} <: Tree{L}
  label :: L
  left  :: Tree{L}
  right :: Tree{L}
end
struct Leaf{L} <: Tree{L}
  label :: L
end

Base.map(f, tree::Leaf) = Leaf(f(tree.label))
Base.map(f, tree::Binary) = Binary(f(tree.label), map(f, tree.left), map(f, tree.right))

function dict2tree(f, dict)
  if isempty(dict["children"])
    Leaf{String}(f(dict["label"]))
  else
    @assert length(dict["children"]) == 2
    Binary{String}(
      f(dict["label"]), 
      dict2tree(f, dict["children"][1]), 
      dict2tree(f, dict["children"][2]) )
  end
end

dict2tree(dict) = dict2tree(identity, dict)

function innerlabels(tree::Tree{L}) where L
  labels = L[]

  pushlabels(tree::Binary) = begin
    push!(labels, tree.label)
    pushlabels(tree.left)
    pushlabels(tree.right)
  end
  pushlabels(::Leaf) = nothing

  pushlabels(tree)
  labels
end

function leaflabels(tree::Tree{L}) where L
  labels = L[]

  pushlabels(tree::Binary) = (pushlabels(tree.left); pushlabels(tree.right))
  pushlabels(tree::Leaf) = push!(labels, tree.label);

  pushlabels(tree)
  labels
end

function relabel_with_spans(tree)
  k = 0 # leaf index
  next_leafindex() = (k += 1; k)
  span(i, j) = (from=i, to=j)
  combine(span1, span2) = span(span1.from, span2.to)

  function relabel(tree::Leaf) 
    i=next_leafindex()
    Leaf(span(i,i))
  end
  function relabel(tree::Binary) 
    left = relabel(tree.left)
    right = relabel(tree.right)
    Binary(combine(left.label, right.label), left, right)
  end

  relabel(tree)
end

constituent_spans(tree) = tree |> relabel_with_spans |> innerlabels

function tree_similarity(tree1, tree2)
  spans1 = constituent_spans(tree1)
  spans2 = constituent_spans(tree2)
  @assert length(spans1) == length(spans2)
  length(intersect(spans1, spans2)) / length(spans1)
end

#############
### Rules ###
#############

struct CFRule{T}
  lhs :: T
  rhs :: Tuple{Vararg{T}}
end

CFRule(lhs, rhs...) = CFRule(lhs, rhs)

# left-most derivation
function derivation(tree::Tree{L}) where L
  rules = CFRule{L}[]

  push_rules(tree::Binary) = begin
    push!(rules, CFRule(tree.label, (tree.left.label, tree.right.label)))
    push_rules(tree.left)
    push_rules(tree.right)
  end
  push_rules(tree::Leaf) = push!(rules, CFRule(tree.label, tree.label))

  push_rules(tree)
  rules  
end

function derivation2tree(derivation::Vector{CFRule{T}}) where T
  i = 0 # rule index
  next_rule() = (i += 1; derivation[i])
  
  function rewrite(nt)
    rule = next_rule()
    @assert nt == rule.lhs && 1 <= length(rule.rhs) <= 2
    if length(rule.rhs) == 1 # terminal rule
      @assert rule.lhs == rule.rhs[1]
      Leaf(nt)
    else # binary rule
      Binary(nt, rewrite(rule.rhs[1]), rewrite(rule.rhs[2]))
    end
  end

  rewrite(first(derivation).lhs)
end

################
### Grammars ###
################

struct Grammar{T}
  rule_logprobs :: Dict{CFRule{T}, Float64}
end

function logprob(grammar::Grammar{T}, rule::CFRule{T}) where T
  LogProb(grammar.rule_logprobs[rule], islog=true)
end

function rules(::Grammar{T}, terminal::T) where T
  [CFRule(terminal, terminal)]
end

function rules(::Grammar{T}, rhs1::T, rhs2::T) where T
  if rhs1 == rhs2
    [CFRule(rhs1, rhs1, rhs2)]
  else
    [CFRule(rhs1, rhs1, rhs2), CFRule(rhs2, rhs1, rhs2)]
  end
end

function treebank_grammar(treebank, all_chords)
  rule_counts = counter(rule for tune in treebank for rule in derivation(tune.tree))
  smoothed_binary_rule_counts = Dict(
    CFRule(c, d, e) => 0.01 + rule_counts[CFRule(c, d, e)]
    for c in all_chords for d in all_chords for e in all_chords
    if c == d || c == e )
  smoothed_terminal_rule_counts = Dict(
    CFRule(c, c) => 0.01 + rule_counts[CFRule(c, c)]
    for c in all_chords )
  smoothed_rule_counts = merge(smoothed_binary_rule_counts, smoothed_terminal_rule_counts)
  smoothed_lhs_counts = Dict(
    chord => sum(count for (rule, count) in smoothed_rule_counts if rule.lhs == chord)
    for chord in all_chords )
  rule_logprobs = Dict(
    rule => log(count) - log(smoothed_lhs_counts[rule.lhs])
    for (rule, count) in smoothed_rule_counts )
  Grammar(rule_logprobs)
end

function parse_chart(grammar, score, terminalss)
  n = length(terminalss) # sequence length
  chart = Dict((i, i) => Dict(t => sum(score(grammar, rule) for rule in rules(grammar, t)) for t in ts)
               for (i, ts) in enumerate(terminalss) )

  for l in 1:n-1 # length
    for i in 1:n-l # start index
      j = i + l # end index
      cell = valtype(chart)()
      for k in i:j-1 # split index
        for (nt1, s1) in chart[i,k]
          for (nt2, s2) in chart[k+1,j]
            for rule in rules(grammar, nt1, nt2)
              s = score(grammar, rule)
              if haskey(cell, rule.lhs)
                cell[rule.lhs] += s * s1 * s2
              else
                cell[rule.lhs] = s * s1 * s2
              end
            end
          end
        end
      end
      chart[i,j] = cell
    end
  end

  chart
end

##############
### Scores ###
##############

inside_score(grammar, rule) = logprob(grammar, rule)

struct BestDerivation{T}
  logprob :: LogProb
  rules   :: Vector{CFRule{T}}
end

+(d1, d2) = d1.logprob > d2.logprob ? d1 : d2
*(d1, d2) = BestDerivation(d1.logprob * d2.logprob, [d1.rules; d2.rules])

best_derivation_score(grammar, rule) = BestDerivation(logprob(grammar, rule), [rule])

#####################################
### Uniformly Random Binary Trees ###
#####################################

catalan(n) = binomial(BigInt(2n), n) - binomial(BigInt(2n), n+1)
@memoize num_trees(num_leafs) = Float64(catalan(num_leafs-1))

function rand_tree(num_leafs)
  if num_leafs == 1
    Leaf(nothing)
  else
    n = num_leafs
    probs = [num_trees(k) * num_trees(n-k) / num_trees(n) for k in 1:n-1]
    k = rand(Categorical(probs))
    Binary(nothing, rand_tree(k), rand_tree(n-k))
  end
end

#############
### Utils ###
#############

function title_and_tree(tune)
  remove_asterisk(label::String) = replace(label, "*" => "")
  (title = tune["title"], 
   tree = @_ tune["trees"][1]["open_constituent_tree"] |> dict2tree(remove_asterisk, __))
end

normalize(iter) = collect(iter) ./ sum(iter)

#############
### Tests ###
#############

# for tune in treebank
#   @assert tune.tree |> derivation |> derivation2tree == tune.tree
# end

# for n in 2:100
#   @assert sum(num_trees(k) * num_trees(n-k) for k in 1:n-1) ≈ num_trees(n) 
# end

###################
### Application ###
###################

function prediction_acc(treebank, i)
  all_chords = unique(chord for tune in treebank for chord in leaflabels(tune.tree))
  tree = treebank[i].tree
  g = treebank_grammar([treebank[1:i-1]; treebank[i+1:end]], all_chords)
  terminals = leaflabels(tree)
  chart = parse_chart(g, best_derivation_score, map(t -> [t], terminals))
  prediction = @_ chart[1,length(terminals)] |> 
    values |> sum |> getproperty(__, :rules) |> derivation2tree 
  tree_similarity(tree, prediction)
end

function mean_random_acc(treebank, i; n_samples=10_000)
  tree = treebank[i].tree
  num_leafs = length(leaflabels(tree))
  mean(tree_similarity(tree, rand_tree(num_leafs)) for _ in 1:n_samples)
end

function context_predictions(grammar, all_chords, contexts; head="C^7", max_continuation_length=3)
  @assert 1 == length(unique(length.(contexts.chords)))
  m = length(contexts.chords[1])
  n = m + max_continuation_length

  chart = parse_chart(grammar, inside_score, fill(all_chords, n))
  normalizing_const = sum(chart[1,k][head] for k in m:n)
  @show length_marginals = [chart[1,k][head] for k in m:n] / normalizing_const

  function calculate_predictions(context)
    terminalss = [map(t -> [t], context.chords); fill(all_chords, 3)]
    chart = parse_chart(grammar, inside_score, terminalss)
    joint_probs = [chart[1,k][head] for k in m:n] / normalizing_const
    context_marginal = sum(joint_probs)
    length_posts = joint_probs / context_marginal # == normalize(joint_probs)
    length_lilihds = joint_probs ./ length_marginals
    norm_lilihds = normalize(length_lilihds)

    ( X = context.id 
    , stimulus = context.stimulus
    , context = prod(context.chords .* " ")[1:end-1]
    , posterior0 = float(length_posts[1])
    , posterior1 = float(length_posts[2])
    , posterior2 = float(length_posts[3])
    , posterior3 = float(length_posts[4])
    , normedlilihd0 = float(norm_lilihds[1])
    , normedlilihd1 = float(norm_lilihds[2])
    , normedlilihd2 = float(norm_lilihds[3])
    , normedlilihd3 = float(norm_lilihds[4])
    )
  end

  @showprogress map(calculate_predictions, eachrow(contexts))
end

function run_main()
  # read and transform treebank data
  treebank_url = "https://raw.githubusercontent.com/DCMLab/JazzHarmonyTreebank/master/treebank.json"
  tunes = HTTP.get(treebank_url).body |> String |> JSON.parse
  treebank = @_ tunes |> filter(haskey(_, "trees"), __) |> map(title_and_tree, __)
  all_chords = unique(chord for tune in treebank for chord in leaflabels(tune.tree))
  full_grammar = treebank_grammar(treebank, all_chords)
  # read chord sequence contexts form the experiment
  contexts = select(
    CSV.read(joinpath(@__DIR__, "..", "data", "Stimuli.csv"), DataFrame),
    :X => :id, 
    :Stimulus => :stimulus, 
    :Progression => (strings -> map(Vector{String} ∘ split, strings)) => :chords)
  
  println("\nCalculating treebank prediction accuracies")
  pred_accs = @showprogress [prediction_acc(treebank, i) for i in eachindex(treebank)] # ~40 sec
  println("mean prediction accuracy: ", mean(pred_accs), "\n")

  println("Calculating random baseline accuracies")
  rand_accs = @showprogress [mean_random_acc(treebank, i) for i in eachindex(treebank)] # ~100 sec
  println("mean random baseline accuracy: ", mean(rand_accs), "\n")
  treebank_results = map(zip(treebank, pred_accs, rand_accs)) do (tune, pred_acc, rand_acc)
    chord_sequence = prod(leaflabels(tune.tree) .* " ")[1:end-1]
    (tune.title, pred_acc=pred_acc, rand_acc=rand_acc, chord_sequence)
  end
  CSV.write(joinpath(@__DIR__, "..", "data", "treebank_results.csv"), treebank_results)

  println("Calculating context predictions")
  context_results = context_predictions(full_grammar, all_chords, contexts)
  CSV.write(joinpath(@__DIR__, "..", "data", "context_results.csv"), context_results)
end

end # module
