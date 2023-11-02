# Splitting audio files into valid-length utterances for training.

## Problem

Each training example must be between $p$ and $q$ seconds long.

Audio of speaker-turns need to be split into valid-lengthed training examples, but can sometimes be too short and long. (less than $p$, more than and $q$)

Each speaker-turn can only be split in between words. But not between all words, because the alignment score between some words arent perfect. The alignment score indicates how accruate the start and end time labels of a word are.

Each audio has a set of valid split points $s$ .

Split the audio file into $s + 1$ sub-arrays, so that no two sub-arrays overlap, and each sub-array is between $p$ and $q$ seconds long. But, they sub-arrays do not have to make up contiguous blocks - one sub-array can end at a split point, and the next sequential sub-array can start at a later split point, though this would leave some of the audio with an invalid segment.


### Objective function

...

### Constraints

...

## Problem Extension 1: Some split points are more optimal than others.

Because the split points have alignment scores, these can be used to select better (or worse) split points.

The alignment score can be combined with the duration of silences following the scored word, to give a more useful score for each split point.

The trade off between minimising wasted audio length, and maximising the split point scores can be done with a constant.

> Because the Synthesizer model ([Tacotron](https://arxiv.org/pdf/1703.10135.pdf)) being fine-tuned is a Sequence to sequence model, anomalies in the data are propoated forward through time. So, it's more important to have a sub-array with an accurate start time, than end time. So, a low alignment score preceeding a long silence is not as bad as a low alignmentscore precceding a short silnce (i.e. between words in fast-pace speech)

### Decision Variables:

 * For each array, it has a set of valid split indices $V_i$.
 * For each valid split index, there is a binary decision variable indicating whether or not to include it in the split. The decision variable for split index at $j$ of array $i$ can be denoted $x_{ij}$, where $x_{ij} = 1$ if split index $j$ is chosen, and $x_{ij} = 0$ otherwise.

Let the Cost of a candidate solution be a weighted sum of two parts;
 * $L_{\text{remain}}$ - The sum of remaining audio-time not included in the valid sub-arrrays.
 * $C_{\text{cost}}$ - The optimiality of split points. Low word alignment scores ($1 - c_{ij}$) chosen as split points, would increase this component in the objective function.

$$
C_{\text{cost}} = \sum_{i} \sum_{j \in V_i} c_{ij} \cdot x_{ij}
$$

Where $\alpha$ and $\beta$ are two constants that determine the weighting of the 2 parts.

$$
\text{Total Cost} = \alpha \cdot L_{\text{remain}} + \beta \cdot C_{\text{cost}}
$$

### Constraints

**Sub-arrays must be longer than $p$, and shorter than $q$;**
<!-- $$ p \leq \sum_{j \in V_i} x_{ij} \leq q $$ -->
For two sequential split indices $j$ and $k$;

$$
p \leq k - j \leq q
$$

**No two sequential sub-arrays can be overlapping;**

To ensure sub-arrays arent selected such that they overlap, the following constraint must be satisfied.
<!-- $$
\sum_{j \in V_i} j \cdot x_{ij} + p \cdot (1 - x_{i,j-1}) \leq \sum_{j \in V_i} j \cdot x_{ij} + q \cdot (1 - x_{i,j-1})
$$ -->

$$
\sum_{k = i}^{j} x[k] = 1, \text{for all valid j and k}
$$

Where $j$, and $k$ are the start and end indicies of a sub-array.


This would be framing the it as a combinatorial problem. It's non-linear since the constraints involve differences between indicies. It's non-convext because the decision variables are binary (whrther or not a split point is selected)

## Problem Extension 2: Maximise the normality of the distribution of sub-array lengths.

Over the entire set of training examples from a single speaker, the distribution of sub-array lengths should also be as close to normal as possible.


### Objective function

To optimise for normality of the distribution of sub-array lengths, the third component could be a measure of the Kolmogorov-Smirnov ($D_{\text{ks}}$) statistic for each candidate solution (sub-array split points, $V_i$ across all arrays).


$$
\text{Total Cost} = \alpha \cdot L_{\text{remain}} + \beta \cdot C_{\text{cost}} + \gamma \cdot D_{\text{ks}}
$$

Where $D_{\text{ks}}$ is the maximum of the absolute value of the difference between $F(x)$, the cumulative distribution function of the sub-array lengths, and $G(x)$ is the cdf of the normal distribution;

$$
D_{\text{ks}} = \max_{i} | F(x_i) - G(x_i) |
$$



### Constraints

Same as Problem 1
