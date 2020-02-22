library(spotifyr)
library(dplyr)
library(ggplot2)
library(caret)
library(InformationValue)
library(ROCR)
library(plotly)

# AUTHORIZATIION ----------------------------------------------------------

id <- '#'
secret <- '#'
Sys.setenv(SPOTIFY_CLIENT_ID = id)
Sys.setenv(SPOTIFY_CLIENT_SECRET = secret)
access_token <- get_spotify_access_token()

# LOADING SADNESS ---------------------------------------------------------

playlist_username <- 'Spotify'
playlist_uri <- '37i9dQZF1DX7qK8ma5wgG1'

SadSongs_features <-
  get_playlist_audio_features(playlist_username, playlist_uri)
SadSongs <-
  as.data.frame(SadSongs_features %>% left_join(get_playlist_tracks(playlist_uri), by = 'track.uri'))

playlist_uri = '37i9dQZF1DX64Y3du11rR1'

SadCovers_features <-
  get_playlist_audio_features(playlist_username, playlist_uri)
SadCovers <-
  as.data.frame(SadCovers_features %>% left_join(get_playlist_tracks(playlist_uri), by = 'track.uri'))

playlist_uri = '37i9dQZF1DWZUAeYvs88zc'

SadBops_features <-
  get_playlist_audio_features(playlist_username, playlist_uri)
SadBops <-
  as.data.frame(SadBops_features %>% left_join(get_playlist_tracks(playlist_uri), by = 'track.uri'))

playlist_uri = '37i9dQZF1DWVV27DiNWxkR'

SadIndie_features <-
  get_playlist_audio_features(playlist_username, playlist_uri)
SadIndie <-
  as.data.frame(SadIndie_features %>% left_join(get_playlist_tracks(playlist_uri), by = 'track.uri'))

playlist_uri = '37i9dQZF1DWVrtsSlLKzro'

SadBeats_features <-
  get_playlist_audio_features(playlist_username, playlist_uri)
SadBeats <-
  as.data.frame(SadBeats_features %>% left_join(get_playlist_tracks(playlist_uri), by = 'track.uri'))

playlist_uri = '37i9dQZF1DWT0RWUaj5iTz'

SadEmoji_features <-
  get_playlist_audio_features(playlist_username, playlist_uri)
SadEmoji <-
  as.data.frame(SadEmoji_features %>% left_join(get_playlist_tracks(playlist_uri), by = 'track.uri'))

playlist_uri = '37i9dQZF1DXaJZdVx8Fwkq'

SadVibe_features <-
  get_playlist_audio_features(playlist_username, playlist_uri)
SadVibe <-
  as.data.frame(SadVibe_features %>% left_join(get_playlist_tracks(playlist_uri), by = 'track.uri'))

playlist_uri = '37i9dQZF1DX15JKV0q7shD'

Crying_features <-
  get_playlist_audio_features(playlist_username, playlist_uri)
Crying <-
  as.data.frame(Crying_features %>% left_join(get_playlist_tracks(playlist_uri), by = 'track.uri'))

Sad <-
  unique(rbind(
    SadBeats,
    SadBops,
    SadCovers,
    SadEmoji,
    SadIndie,
    SadSongs,
    SadVibe
  ))
Sad$Sad <- '1'

# LOADING HAPPINESS -------------------------------------------------------

playlist_uri = '37i9dQZF1DX84kJlLdo9vT'

HappyDays_features <-
  get_playlist_audio_features(playlist_username, playlist_uri)
HappyDays <-
  as.data.frame(HappyDays_features %>% left_join(get_playlist_tracks(playlist_uri), by = 'track.uri'))

playlist_uri = '37i9dQZF1DXdPec7aLTmlC'

HappyHits_features <-
  get_playlist_audio_features(playlist_username, playlist_uri)
HappyHits <-
  as.data.frame(HappyHits_features %>% left_join(get_playlist_tracks(playlist_uri), by = 'track.uri'))

playlist_uri = '37i9dQZF1DX9u7XXOp0l5L'

HappyTunes_features <-
  get_playlist_audio_features(playlist_username, playlist_uri)
HappyTunes <-
  as.data.frame(HappyTunes_features %>% left_join(get_playlist_tracks(playlist_uri), by = 'track.uri'))

playlist_uri = '37i9dQZF1DWVlYsZJXqdym'

HappyPop_features <-
  get_playlist_audio_features(playlist_username, playlist_uri)
HappyPop <-
  as.data.frame(HappyPop_features %>% left_join(get_playlist_tracks(playlist_uri), by = 'track.uri'))

playlist_uri = '37i9dQZF1DWSf2RDTDayIx'

HappyBeats_features <-
  get_playlist_audio_features(playlist_username, playlist_uri)
HappyBeats <-
  as.data.frame(HappyBeats_features %>% left_join(get_playlist_tracks(playlist_uri), by = 'track.uri'))

playlist_uri = '37i9dQZF1DWVOMXLzSabIM'

HappyThrowbacks_features <-
  get_playlist_audio_features(playlist_username, playlist_uri)
HappyThrowbacks <-
  as.data.frame(HappyThrowbacks_features %>% left_join(get_playlist_tracks(playlist_uri), by = 'track.uri'))

playlist_uri = '37i9dQZF1DWZKuerrwoAGz'

HappyFavorites_features <-
  get_playlist_audio_features(playlist_username, playlist_uri)
HappyFavorites <-
  as.data.frame(HappyFavorites_features %>% left_join(get_playlist_tracks(playlist_uri), by = 'track.uri'))

Happy <-
  unique(
    rbind(
      HappyBeats,
      HappyDays,
      HappyFavorites,
      HappyHits,
      HappyPop,
      HappyThrowbacks,
      HappyTunes
    )
  )
Happy$Sad <- '0'

# EXPLORING THE DATA -----------------------------------------------------

songs <-
  as.data.frame(unique(rbind(Happy[, c(6:16, 17, 77, 101)], Sad[, c(6:16, 17, 77, 101)])))
# I pulled 1108 songs, but after filtering by unique track ids (x and y), 1067 unique observations remain.

summary(Happy[, c(6:16)])
summary(Sad[, c(6:16)])
# 76.5% of happy songs in mode 1 compared to 69.29% of sad songs, only a slight difference
# Not a significant difference in the distribution of key usage
ggplot(songs, aes(x = factor(key), fill = Sad)) +
  geom_bar() +
  theme_minimal() +
  xlab('Key') +
  ylab('Count') +
  scale_fill_discrete(name = "Mood", labels = c('Happy', 'Sad'))

# Minumum and 1Q danceability scores noticably different
ggplot(songs, aes(x = danceability, fill = Sad)) +
  geom_density(alpha = 0.7) +
  theme_minimal() +
  xlab('Danceability') +
  ylab('Density') +
  scale_fill_discrete(name = "Mood", labels = c('Happy', 'Sad')) # Plot reveals there's a lot of overlap outside of extremes in happy

# Liveness 3Q and max scores also evidently different
ggplot(songs, aes(x = liveness, fill = Sad)) +
  geom_density(alpha = 0.7) +
  theme_minimal() +
  xlab('Liveness') +
  ylab('Density') +
  scale_fill_discrete(name = "Mood", labels = c('Happy', 'Sad')) # Once again, lots of overlap - just an obvious spike in sad

# I also want to see the valence distribution since that's supposed to be a good predicter of a songs happiness.
ggplot(songs, aes(x = valence, fill = Sad)) +
  geom_density(alpha = 0.7) +
  theme_minimal() +
  xlab('Valence') +
  ylab('Density') +
  scale_fill_discrete(name = "Mood", labels = c('Happy', 'Sad')) # Looks better than some of the other variables, but not perfect

# I'm going to create lists of the artists of the sad and happy songs, just out of curiousity.
# This process is complicated by the fact that this information is stored as a list within a list within the song features.
sad_artists = list()
i <- 1
for (a in 1:nrow(Sad)) {
  item <- Sad$track.artists.x[[a]][[3]]
  l <- length(item)
  for (x in 1:l) {
    sad_artists[i] <- item[[x]]
    i <- i + 1
  }
}

happy_artists = list()
i <- 1
for (a in 1:nrow(Happy)) {
  item <- Happy$track.artists.x[[a]][[3]]
  l <- length(item)
  for (x in 1:l) {
    happy_artists[i] <- item[[x]]
    i <- i + 1
  }
}

sad_artists <- unlist(sad_artists)
sad_artists_df <-
  as.data.frame(sort(table(sad_artists), decreasing = T))
print(sad_artists_df[1:10, ]) # 10 most frequent artists of sad songs

happy_artists <- unlist(happy_artists)
happy_artists_df <-
  as.data.frame(sort(table(happy_artists), decreasing = T))
print(happy_artists_df[1:10, ]) # 10 most frequent artists of happy songs

# MODELING ----------------------------------------------------------------

# I'm going to create a simple logistic regression to predict the probability of a song being sad.
# Yes, I could just as easily train a Random Forest or Gradient Boosting Classifier, but I like to start simple first.
# If necessary, if I'm unhappy with the performance of this model, I'll try more advanced methods.

target <- songs[, 14] # Sad
features <- songs[, 1:11]  # Song features

# Key and Mode are factors, so I'm going to create dummy variables for each.
features$key <- as.factor(features$key)
features$mode <- as.factor(features$mode)
dummy <- dummyVars(" ~.", features)
features <- as.data.frame(predict(dummy, features))

# I need to check the variance of variables to ensure I don't have any potential complete or partial seperation issues.
nzv <-
  nearZeroVar(features, saveMetrics = TRUE) # Reveals near zero variance in variables key.3, key.10
data <- cbind(features, target)
table(data$target, data$key.3)
table(data$target, data$key.10)
# After confirming less than 5% of observations fall into each of these categories, I've decided to combine key.3 and key.10

data$key.3.10 <- data$key.3 + data$key.10
table(data$target, data$key.3.10)
# The new level accounts for 8% of the observations, which should be fine. I'll now drop key.3 and key.10.
# I'm also going to drop that mode.0 variable because this is a binary factor not requiring 2 dummy columns.

data <- data[,-c(6, 13, 16)]

# Now I'm going to split the data into training and testing subsets.
train_index <- createDataPartition(data$target,
                                   p = .7,
                                   list = FALSE,
                                   times = 1)

train <- as.data.frame(data[train_index, ])
test <- as.data.frame(data[-train_index, ])

# I'm going to use backward selection for feature selection.
model <-
  glm(
    target ~ danceability + energy + factor(key.0) + factor(key.1) +
      factor(key.2) + factor(key.4) + factor(key.5) + factor(key.6) +
      factor(key.7) + factor(key.8) + factor(key.9) + factor(key.3.10) +
      factor(key.11) + loudness + factor(mode.1) + speechiness +
      acousticness + instrumentalness + liveness + valence + tempo,
    data = train,
    family = 'binomial'
  )

bmodel <- step(model, direction = "backward")
summary(bmodel)

# Selected variables are energy, key5, key.7, key.9, loudness, mode.1, speechiness, acousticness, liveness, and valence.
# I'd also like to try forard selection to include additional variables of interactions for the continuous variables.

imodel <-
  glm(
    target ~ energy + factor(key.5) + factor(key.7) + factor(key.9) + loudness +
      factor(mode.1) + speechiness + acousticness + liveness + valence +
      danceability * energy + danceability * loudness + danceability * speechiness +
      danceability * acousticness + danceability * instrumentalness +
      danceability * liveness + danceability * valence + danceability *
      tempo +
      energy * loudness + energy * speechiness + energy * acousticness + energy *
      instrumentalness +
      energy * liveness + energy * valence + energy * tempo + loudness *
      speechiness + loudness * acousticness +
      loudness * instrumentalness + loudness * liveness + loudness * valence + loudness *
      tempo +
      speechiness * acousticness + speechiness * instrumentalness +
      speechiness * liveness + speechiness * valence + speechiness * tempo +
      acousticness * instrumentalness + acousticness * liveness + acousticness *
      valence + acousticness * tempo +
      instrumentalness * liveness + instrumentalness * valence + instrumentalness *
      tempo +
      liveness * valence + liveness * tempo + valence * tempo,
    data = train,
    family = 'binomial'
  )

fmodel <- step(bmodel,
               scope = list(lower = formula(model), upper = formula(imodel)),
               direction = "forward")
summary(fmodel)

# AIC is slightly better, so I'm going to proceed with this model.
# The additional interaction variables are loudness*valence, energy*liveness, speechiness*liveness, and speechiness*acousticness.

plot(fmodel, 4) # Cook's distance plot; I see a few outliers but no alarming patterns.

# I'm going to plot the discrimination slope.
train$p_hat <-
  predict(fmodel, type = "response") # Predicted probabilities for training data
p1 <- train$p_hat[train$target == 1]
p0 <- train$p_hat[train$target == 0]
coef_discrim <- mean(p1) - mean(p0) # 0.645434569

ggplot(train, aes(x = p_hat, fill = target)) +
  geom_density(alpha = 0.7) +
  scale_fill_grey() +
  labs(x = "Predicted Probability",
       y = "Density",
       fill = "Outcome",
       title = "Training Predictions") +
  theme_minimal()

# Let's look at the percent concordance and plot the ROC curve.
Concordance(train$target, train$p_hat) # Concordance percentage is 0.9475394 on training data
plotROC(train$target, train$p_hat)
AUROC(train$target, train$p_hat) # Not bad!

# I'm going to take a look at KS stats for this model.
ks_plot(train$target, train$p_hat)
ks_stat(train$target, train$p_hat)

pred <- prediction(fitted(fmodel), factor(train$target))
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
KS <- max(perf@y.values[[1]] - perf@x.values[[1]])
cutoffAtKS <-
  unlist(perf@alpha.values)[which.max(perf@y.values[[1]] - perf@x.values[[1]])]
print(c(KS, cutoffAtKS)) # KS: 0.7687774 Cut-off: 0.5066909 (on training data)

# Now I'm going to score the test data.
test$p_hat <-
  predict(fmodel, newdata = test[, 1:21], type = 'response')

Concordance(test$target, test$p_hat) # Concordance percentage is 0.9305818 on testing data
# This value is not far from the percent concordance on the training data, so I don't think there're any overfitting issues.

confusionMatrix(test$target, test$p_hat, threshold = 0.5) # Using the cut-off from the KS test, ~86% accuracy!
sensitivity(test$target, test$p_hat, threshold = 0.5)
specificity(test$target, test$p_hat, threshold = 0.5)

# This creates a visual distribution of the predicted probabilities and actual target.
ggplot(test, aes(x = p_hat, fill = target)) +
  geom_density(alpha = 0.7) +
  theme_minimal()

# VISUALS -----------------------------------------------------------------

# The most significant continuous variables in this model (according to their p-values)
# are energy, loudness*valence, and energy*liveness, so I'm going to create a 3D plot of these.

p <-
  plot_ly(
    songs,
    x = ~ energy,
    y = ~ loudness*valence,
    z = ~ energy*liveness,
    color = ~ Sad,
    colors = c('Orange', 'Blue')
  ) %>%
  add_markers() %>%
  layout(scene = list(
    xaxis = list(title = 'Energy'),
    yaxis = list(title = 'Loudness x Valence'),
    zaxis = list(title = 'Energy x Liveness')
  ))
p # Pretty clear clusters but still some obvious overlap

# Let's try speechiness, acousticness, and loudness — these are the least significant continuous variables by p-values.

q <-
  plot_ly(
    songs,
    x = ~ speechiness,
    y = ~ loudness,
    z = ~ liveness,
    color = ~ Sad,
    colors = c('Orange', 'Blue')
  ) %>%
  add_markers() %>%
  layout(scene = list(
    xaxis = list(title = 'Speechiness'),
    yaxis = list(title = 'Acousticness'),
    zaxis = list(title = 'Loudness')
  ))
q # Definitely less of a clear spread here.

# RESULT EXPLORATION ------------------------------------------------------

filt_songs <- songs[, c(2, 36, 6:16, 101)]

# I'm going to use the model to score the whole dataset now, because I want to get an idea of what might be the
# most likely to be sad and most likely to be happy songs. I also want to see where the model is going wrong.

info <- filt_songs[, c(1, 2, 14)] # Sad, track name, playlist name
all_features <- filt_songs[, c(3:13)]

# Key and Mode are factors, so I'm going to create dummy variables for each.
all_features$key <- as.factor(all_features$key)
all_features$mode <- as.factor(all_features$mode)
dummy <- dummyVars(" ~.", all_features[, 1:11])
all_features <- as.data.frame(predict(dummy, all_features))

all_filt_songs <- cbind(info, all_features[, c(1:5, 7:15, 17:23)])

all_filt_songs$p_hat <-
  predict(fmodel, all_filt_songs[, 4:24], type = "response")

# Highest probability: "Stay With Me - Live From Spotify Berlin" @ 0.9999954
print(all_filt_songs[all_filt_songs$p_hat == max(all_filt_songs$p_hat), ])

# Lowest probability: "Just Can't Get Enough - Live in Hammersmith" @ 2.550584e-05
print(all_filt_songs[all_filt_songs$p_hat == min(all_filt_songs$p_hat), ])

# Mean probability playlist by playlist:
print(all_filt_songs %>%
        group_by(playlist_name) %>%
        summarise(mean(p_hat)))
# Lowest mean probability: Happy Favorites
# Highest mean probability: Sad Covers

# Now let's look at what the model is getting wrong!
oops_happy <- all_filt_songs %>%
  filter(Sad == 0 &
           p_hat >= 0.5) # 58 songs that should have had higher probabilities
oops_sad <- all_filt_songs %>%
  filter(Sad == 1 &
           p_hat <= 0.5) # 75 songs that should have had lower probabilities
