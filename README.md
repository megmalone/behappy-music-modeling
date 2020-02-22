# behappy-music-modeling

## Background
The goal of this project was to devise a way that analytics might be used to improve the Spotify user experience. I decided to build a model capable of differentiating between sad and happy songs with the thinking that it could be used to develop an application to identify when Spotify users are listening to many sad songs in a row — I call this hypothetical application "behappy." The application might then recommend happier songs for the users queue in an effort to cheer them up. 

The current default measure of a songs' happiness or lack thereof is valence. According to the Spotify API documentation, "[The valence variable is a] measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry)." This is not a perfect predictor, however, and that is why a model such as this would be necessary to develop an application that could identify sad songs more accurately.

## Model
I trained and tested a model to predict whether or not a song was sad with the features of 1,067 songs downloaded with the Spotify API. Songs were selected from several playlists curated by Spotify and dedicated to happy (e.g. Happy Throwbacks, Happy Hits, Happy Favorites) or sad songs (e.g. Sad Bops, Sad Covers, Sad Indie). The logistic regression model I developed was 86% accurate when scoring the testing data subset with 93% concordance. The variables in this model include:
* energy
* key.5 (a binary dummy variable)
* key.7 (a binary dummy variable)
* key.9 (a binary dummy variable)
* loudness
* mode.1 (a binary dummy variable)
* speechiness
* acousticness
* liveness
* valence
* loudness x valence
* speechiness x liveness
* speechiness x acousticness

## Process
I began by using backward selection on the songs features available through the Spotify API without interactions. Then, I used forward selection to evaluate the addition of interaction variables. The resulting model included the variables listed above. The KS statistic on the training data was 0.77 with a cut-off of 0.5. This cut-off was used as the threshold to determine the assignment of predicted outcomes. As also mentioned about, the percent concordance on the testing data was 0.93, with a sensitivity of 0.85 and specificity of 0.86.

## Next Steps
While reviewing the results, I did notice that although duplicate track IDs were initially filtered out, some duplicated songs remained in the data this model was trained and tested on. Some of these were perfect duplicates — all of their features matched — and I suspect that these were not eliminated because they came from different albums and therefore had different IDs. Other duplicates, while very similary, had slightly different features, and I suspect that they were covers or different recordings of the same songs. It is unknown how these duplications may have affected the model and could be investigated further.

I would in the future like to try different machine learning algorithms such as a Random Forest or Gradient Boosting Classifier. For times sake and for the purpose of demonstrating multiple skills, I chose the simpler logistic regression approach for this project.

I would also like to try developing a similer multinomial classification model with a variety of different moods such as angry, depressed, etc. The challenge with that task would be finding reliable playlists to pull identified songs from.

## References
LOGIT REGRESSION | R DATA ANALYSIS EXAMPLES: https://stats.idre.ucla.edu/r/dae/logit-regression/
The caret Package: http://topepo.github.io/caret/model-training-and-tuning.html#extracting-predictions-and-class-probabilities
