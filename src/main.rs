use std::collections::{HashMap, HashSet};
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
    let badgers = reader.deserialize().collect::<Result<Vec<_>, _>>()?;

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

struct Generation<'a> {
    pub badgers: &'a Vec<Badger>,
    pub population: Vec<Vec<usize>>,
    pub ngroups: usize,
    pub crossover_rate: f64,
    pub mutation_rate: f64,
    pub survival_rate: f64,
}

impl<'a> Generation<'a> {
    fn new(badgers: &'a Vec<Badger>, population_size: usize, ngroups: usize) -> Self {
        let rng = rand::thread_rng();
        let dist = Uniform::new(1, ngroups);
        let init_pop: Vec<Vec<usize>> = (0..population_size)
            .map(|_| rng.sample_iter(dist).take(badgers.len()).collect())
            .collect();

        assert_eq!(init_pop.len(), population_size);

        Generation {
            badgers,
            population: init_pop,
            ngroups,
            crossover_rate: 0.85,
            mutation_rate: 0.15,
            survival_rate: 0.2,
        }
    }

    pub fn next_gen(&self) -> Self {
        let size = self.population.len();

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

        Generation {
            badgers: self.badgers,
            population,
            ..*self
        }
    }

    fn fittest(&self) -> Vec<Vec<usize>> {
        let mut fittest = self.population.clone();

        // Selection

        fittest.sort_by(|a, b| {
            let sa = Solution::new(a, self.badgers);
            let sb = Solution::new(b, self.badgers);

            sb.score().partial_cmp(&sa.score()).unwrap()
        });

        let survivor_count = (fittest.len() as f64 * self.survival_rate).floor() as usize;
        fittest[0..survivor_count].to_owned()
    }
}

// Fitness

#[derive(Debug)]
struct Solution<'a>(Vec<(&'a usize, &'a Badger)>);

impl<'a> fmt::Display for Solution<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut groups: HashMap<usize, Vec<&Badger>> = HashMap::new();

        for (group, badger) in &self.0 {
            groups.entry(**group).or_insert(vec![]).push(badger);
        }

        let ideal_size = self.0.len() as f64 / groups.len() as f64;
        for (n, group) in &groups {
            write!(
                f,
                "= Group #{} ({} badgers, score: {}): \n",
                n,
                group.len(),
                Solution::group_score(&group, ideal_size)
            )?;

            for badger in group {
                write!(f, "{}\n", badger)?;
            }

            write!(f, "\n")?;
        }
        write!(f, "Total score: {}\n\n", self.score())?;

        Ok(())
    }
}

impl<'a> Solution<'a> {
    pub fn new(groups: &'a Vec<usize>, badgers: &'a Vec<Badger>) -> Self {
        Solution(groups.iter().zip(badgers.iter()).collect())
    }

    pub fn score(&self) -> f64 {
        let mut groups: HashMap<usize, Vec<&Badger>> = HashMap::new();

        for (group, badger) in &self.0 {
            groups.entry(**group).or_insert(vec![]).push(badger);
        }

        let ideal_size = self.0.len() as f64 / groups.len() as f64;
        groups
            .values()
            .map(|g| Solution::group_score(g, ideal_size))
            .sum()
    }

    fn group_score(group: &Vec<&Badger>, ideal_size: f64) -> f64 {
        let mut genders: HashSet<&String> = HashSet::new();
        let mut disciplines: HashSet<&String> = HashSet::new();
        let mut seniorities: HashSet<&String> = HashSet::new();
        let mut clients: HashSet<&String> = HashSet::new();
        let mut teams: HashSet<&String> = HashSet::new();

        for badger in group {
            genders.insert(&badger.gender);
            disciplines.insert(&badger.discipline);
            seniorities.insert(&badger.seniority);
            clients.insert(&badger.client);
            teams.insert(&badger.team);
        }

        let size = -(ideal_size - group.len() as f64).abs();
        let genders: f64 = disciplines.len() as f64;
        let disciplines: f64 = disciplines.len() as f64;
        let seniorities: f64 = seniorities.len() as f64;
        let clients: f64 = clients.len() as f64;
        let teams: f64 = teams.len() as f64;

        4.0 * size + 5.0 * genders + disciplines + seniorities + clients + teams
    }
}

fn main() {
    match read_badgers() {
        Err(err) => {
            println!("Could not read badgers: {}", err);
            process::exit(1);
        }
        Ok(badgers) => {
            let mut generation = Generation::new(&badgers, 60, 9);

            generation.crossover_rate = 0.6;
            generation.mutation_rate = 0.5;
            generation.survival_rate = 0.3;

            for i in 0..150 {
                let fittest = generation.fittest();
                let best = &fittest[0];
                println!(
                    "Gen {} - best: {}",
                    i,
                    Solution::new(best, &badgers).score()
                );

                generation = generation.next_gen();
            }

            let fittest = generation.fittest();
            let best = &fittest[0];
            println!("Winner:\n{}", Solution::new(best, &badgers));
        }
    }
}
