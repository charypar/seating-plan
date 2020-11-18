use std::collections::HashMap;
use std::error::Error;
use std::{fmt, io, process};

use csv;
use rand::{self, distributions::Distribution, distributions::Uniform, Rng};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Badger {
    pub name: String,
    pub gender: String,
    pub discipline: String,
    pub seniority: String,
    pub client: String,
    pub team: String,
}

fn read_badgers() -> Result<Vec<Badger>, Box<dyn Error>> {
    let mut reader = csv::Reader::from_reader(io::stdin());
    let badgers = reader.deserialize().collect::<Result<Vec<Badger>, _>>()?;

    Ok(badgers)
}

impl fmt::Display for Badger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({}, {}, {}, {}, {})",
            self.name, self.gender, self.discipline, self.seniority, self.client, self.team
        )?;

        Ok(())
    }
}

// Genetic optimization

fn mutate<D>(individual: &Vec<usize>, dist: D) -> Vec<usize>
where
    D: Distribution<usize>,
{
    let mut rng = rand::thread_rng();
    let idx = rng.gen_range(0, individual.len());
    let value = rng.sample(dist);

    let mut result = individual.clone();
    result[idx] = value;

    result
}

fn cross_over(mother: &Vec<usize>, father: &Vec<usize>) -> Vec<usize> {
    let mut child = Vec::new();

    let mut rng = rand::thread_rng();
    let crossover_point = rng.gen_range(0, mother.len());

    child.extend_from_slice(&mother[0..crossover_point]);
    child.extend_from_slice(&father[crossover_point..]);

    child
}

struct Generation<F: Fn(&Vec<usize>) -> f64> {
    pub population: Vec<Vec<usize>>,
    pub ngroups: usize,
    pub crossover_rate: f64,
    pub mutation_rate: f64,
    pub survival_rate: f64,
    pub fitness: F,
}

impl<F: Fn(&Vec<usize>) -> f64> Generation<F> {
    fn new(population_size: usize, ngroups: usize, nbadgers: usize, fitness: F) -> Self {
        let rng = rand::thread_rng();

        let dist = Uniform::new(1, ngroups);
        let init_pop: Vec<Vec<usize>> = (0..population_size)
            .map(|_| rng.sample_iter(dist).take(nbadgers).collect())
            .collect();

        assert_eq!(init_pop.len(), population_size);

        Generation {
            population: init_pop,
            ngroups,
            crossover_rate: 0.85,
            mutation_rate: 0.15,
            survival_rate: 0.2,
            fitness: fitness,
        }
    }

    pub fn next_gen(self) -> Self {
        let size = self.population.len();

        // Selection

        let fittest = self.fittest();
        let fittest_count = fittest.len();

        // Breeding

        let mut population: Vec<Vec<usize>> = Vec::with_capacity(size);
        let mut rng = rand::thread_rng();

        for _ in 0..size {
            let mother = rng.gen_range(0, fittest_count);

            // Cross-over
            if rng.gen_bool(self.crossover_rate) {
                let father = rng.gen_range(0, fittest_count);

                population.push(cross_over(&fittest[mother], &fittest[father]));
            } else {
                population.push(fittest[mother].clone());
            }

            // Mutation

            if rng.gen_bool(self.mutation_rate) {
                *population.last_mut().unwrap() =
                    mutate(population.last().unwrap(), Uniform::new(0, self.ngroups));
            }
        }

        assert_eq!(self.population.len(), population.len());

        Generation { population, ..self }
    }

    fn fittest(&self) -> Vec<Vec<usize>> {
        let mut fittest = self.population.clone();

        fittest.sort_by(|a, b| (self.fitness)(a).partial_cmp(&(self.fitness)(b)).unwrap());

        let survivor_count = (fittest.len() as f64 * self.survival_rate).floor() as usize;
        fittest[0..survivor_count].to_owned()
    }
}

// Fitness metric calculations

// Histogram keeps representation and proportion of values
#[derive(Debug)]
struct Histogram<T: Eq + std::hash::Hash> {
    counts: HashMap<T, usize>,
    total: usize,
}

impl<T> Histogram<T>
where
    T: Eq + std::hash::Hash,
{
    fn new() -> Self {
        Self {
            counts: HashMap::new(),
            total: 0,
        }
    }

    fn insert(&mut self, item: T) {
        *self.counts.entry(item).or_insert(0) += 1;
        self.total += 1;
    }

    // Distance between two representations
    fn diff(&self, other: &Histogram<T>) -> f64 {
        let diff = (self.counts.len() as i32 - other.counts.len() as i32).abs();

        if diff > 0 {
            return diff as f64;
        }

        // If we we've matched the sets, compare proportions

        let mut score = 0.0;

        for key in self.counts.keys() {
            let self_score = self
                .counts
                .get(key)
                .map(|it| *it as f64 / self.total as f64)
                .unwrap_or(0.0);
            let other_score = other
                .counts
                .get(key)
                .map(|it| *it as f64 / self.total as f64)
                .unwrap_or(0.0);

            score += (self_score - other_score).powf(2.0);
        }

        score
    }
}

// Profile of a group as histograms of attributes
#[derive(Debug)]
struct Profile<'a> {
    genders: Histogram<&'a String>,
    disciplines: Histogram<&'a String>,
    seniorities: Histogram<&'a String>,
    clients: Histogram<&'a String>,
    teams: Histogram<&'a String>,
    pub count: f64,
}

impl<'a> Profile<'a> {
    fn new() -> Self {
        Self {
            genders: Histogram::new(),
            disciplines: Histogram::new(),
            seniorities: Histogram::new(),
            clients: Histogram::new(),
            teams: Histogram::new(),
            count: 0.0,
        }
    }

    fn insert(&mut self, badger: &'a Badger) {
        self.count += 1.0;
        self.genders.insert(&badger.gender);
        self.disciplines.insert(&badger.discipline);
        self.seniorities.insert(&badger.seniority);
        self.clients.insert(&badger.client);
        self.teams.insert(&badger.team);
    }
}

// Fitness score itself, lower is better

fn fitness(solution: &Vec<usize>, badgers: &Vec<Badger>, ideal: &Profile) -> f64 {
    let mut profiles: HashMap<usize, Profile> = HashMap::new();

    for i in 0..solution.len() {
        profiles
            .entry(solution[i])
            .or_insert(Profile::new())
            .insert(&badgers[i])
    }

    profiles
        .values()
        .map(|group| {
            let size = (ideal.count - group.count).abs();

            let gender = ideal.genders.diff(&group.genders);
            let discipline = ideal.disciplines.diff(&group.disciplines);
            let seniority = ideal.seniorities.diff(&group.seniorities);
            let client = ideal.clients.diff(&group.clients);
            let team = ideal.teams.diff(&group.teams);

            10.0 * size + 6.0 * gender + 3.0 * discipline + seniority + 2.0 + client + team
        })
        .sum::<f64>()
}

fn main() {
    match read_badgers() {
        Err(err) => {
            println!("Could not read badgers: {}", err);
            process::exit(1);
        }
        Ok(badgers) => {
            let mut ideal = Profile::new();
            for badger in badgers.iter() {
                ideal.insert(badger)
            }
            ideal.count = badgers.len() as f64 / 9.0;

            // Initial generation

            let mut generation = Generation::new(150, 9, badgers.len(), |solution| {
                fitness(solution, &badgers, &ideal)
            });

            // metaheuristic parameters

            generation.crossover_rate = 0.5;
            generation.mutation_rate = 0.5;
            generation.survival_rate = 0.2;

            // Optimisation loop
            for i in 0..300 {
                let fittest = generation.fittest();
                let best = &fittest[0];
                let score = fitness(best, &badgers, &ideal);

                println!("Gen {:>4} - best: {:.5} - {:?}", i, score, best);

                generation = generation.next_gen();
            }

            // Print results

            let fittest = generation.fittest();
            let best = &fittest[0];

            let mut tagged: Vec<_> = best.into_iter().zip(&badgers).collect();
            tagged.sort_by(|(a, _), (b, _)| a.cmp(b));

            let mut group: i32 = 0;
            for (g, badger) in tagged {
                if *g as i32 > group {
                    group += 1;
                    println!("= Group #{}", group);
                }
                println!("{}", badger);
            }
        }
    }
}
