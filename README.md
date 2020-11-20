# Seating plan

Optimising a seating plan for maximum diversity with genetic algorithms.

## About

This tool atempts to find N optimal subgroups of a larger group, such that various traits are represented fairly in the subgroups.

It uses a [Genetic Algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm) optimisation to find the solution, by evaluating each group and adding up the group scores to the total score of the solution - its "fitness".

A group scores well if:

- Its size is close to the ideal size of `#individuals / #groups`
- For each trait, all the values are represented in the group
  - When all values are represented, they are close to their proportions of the entire population

For example, if there are three different discplines reperesented in the whole group, and we're forming groups of 10, then a group scores better if more disciplines are represented and even better if the three disciplines are represented with the same proportion as in the entire population.

_NOTE: this isn't yet generalised to process any CSV format._

## Usage

Run passing a CSV with the following header into `stdin`:

```csv
name,gender,discipline,seniority,client,team
```

e.g.

```sh
seating-plan < people.csv
```

## How it works

Each item in the list gets assigned a numerical group label. The vector of labels forms a potential solution to the problem.

At the start, we generate a number of random solutions - the initial population. The "fitness" metric is calculated for each solution and the fittest portion of the population is **selected** for "breeding". (The proportion is an input parameter).

Then a new generation is repeatedly created by combining individuals from the previous generation breeding group - either keeping them, or creating a new individual by a **crossover** of two individuals: randoml cuting both solutions in half at the same point and swapping one of the halves. Then some of the individuals are **mutated** by changing one item in the solution to a different number (effectivelly moving one of the people to a different group). Both crossover and mutation rates are also input parameters.
