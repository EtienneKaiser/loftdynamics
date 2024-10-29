# Maneuver Detection for Loft Dynamics

The partner of this project is the Dübendorf-based company Loft Dynamics, 
who develops and produces the world’s only EASA- and FAA-qualified VR helicopter pilot training simulators.
During a training session on the simulator, a helicopter pilot flies a series of different maneuvers, 
such as hovering, climbs, descents or turns. To analyze maneuvers in more detail – either manually by an instructor, 
or automatically using a software – all maneuvers during a flight have to be detected. So far, 
Loft Dynamics uses a hand-crafted rule-based algorithm to detect these maneuvers. 
However, this solution has a series of downsides: it is difficult to extend these rules to new aircraft, 
new maneuvers, or fix problems in the detection. 
In this project, our task is to develop a machine learning (ML) based algorithm which solves these issues 
by learning from past data of maneuvers.
## Caching Mechanism

When you run the program for the first time, it will iterate through all the JSON and Parquet files 
in the `data` directory. This process may take some time as it reads and processes a large amount of data, 
resulting in a cache file being created in the `cachedir` directory. The cache file can be quite large, 
with an expected size of approximately **31 GB**. In subsequent runs, the program will load data from this cache file, 
significantly reducing loading time. The main benefit is that we can work with all data anytime!

## Usage

1. **First Run**: Expect a longer processing time as it will create the cache file. => Take a coffee ☕!
2. **Subsequent Runs**: The program will quickly load data from the cache file, speeding up our iterative progress.

## Contribution

Contributions are welcome! Please adhere to the following guidelines:

- **Do not push** the `cachedir` to the repository. This directory contains large cache files that are not necessary for version control and may unnecessarily inflate the repository size.
- When creating a Pull Request, please name it using the following format:
`feature/LODY-{incrementing number}-{descriptive-branch-name}`