library(ggplot2)
library(dplyr)

dat <- read.csv("../data/benchmark_MIRAGE.csv")

# Count the number of correct answers
total_answers <- nrow(dat)
correct_answers <- sum(dat$correct == 1)
accuracy <- correct_answers / total_answers
print(paste("Total answers:", total_answers))
print(paste("Correct answers:", correct_answers))
print(paste("Accuracy:", accuracy))

# Plot the time it takes to generate the query embedding
p <- ggplot(dat, aes(x = dataset, y = embedding_time, fill=dataset)) +
  geom_boxplot() +
  xlab("Dataset") +
  ylab("Embedding Time (s)") +
  theme_bw() +
  theme(legend.position = "none")

ggsave("../data/embedding_time_boxplot.pdf", plot = p, width = 4, height = 2.5)

# Plot the time it takes to select the relevant data sources
p <- ggplot(dat, aes(x = dataset, y = selection_time, fill=dataset)) +
  geom_boxplot() +
  xlab("Dataset") +
  ylab("DS Selection Time (s)") +
  theme_bw() +
  theme(legend.position = "none")

ggsave("../data/ds_selection_time_boxplot.pdf", plot = p, width = 4, height = 2.5)

# Plot the time it takes for the LLM to generate the final answer
p <- ggplot(dat, aes(x = dataset, y = generate_time, fill=dataset)) +
  geom_boxplot() +
  xlab("Dataset") +
  ylab("LLM Generation Time (s)") +
  theme_bw() +
  theme(legend.position = "none")

ggsave("../data/llm_time_boxplot.pdf", plot = p, width = 4, height = 2.5)

# Plot E2E times
p <- ggplot(dat, aes(x = dataset, y = e2e_time, fill=dataset)) +
  geom_boxplot() +
  xlab("Dataset") +
  ylab("E2E Time (s)") +
  theme_bw() +
  theme(legend.position = "none")

ggsave("../data/e2e_time_boxplot.pdf", plot = p, width = 4, height = 2.5)

# Plot retrieval time per data source
dat <- read.csv("../data/ds_durations_MIRAGE.csv")
dat <- dat[dat$duration <= 20, ] # Remove outliers

p <- ggplot(dat, aes(x = data_source, y = duration, fill=data_source)) +
  geom_boxplot() +
  xlab("Data Source") +
  ylab("Retrieval time (s)") +
  theme_bw() +
  theme(legend.position = "none")

ggsave("../data/retrieve_time_per_ds_boxplot.pdf", plot = p, width = 4, height = 2.5)