import Cocoa
import CreateML

let data = try MLDataTable(contentsOf: URL(fileURLWithPath: "/Users/lovice/Documents/Projects/TwitterSentiment/twitter-sanders-apple3.csv")) // Can be a JSON format or CSV

let(trainingData, testingData) = data.randomSplit(by: 0.8, seed: 5) // Splitting the exisitng data for training and testing, the randomSplit is only available for CreateML
// Here we are splitting the data to 80% for training randomly


//Creating the Machine learning model by passing in the training data
let sentimentClassifier = try MLTextClassifier(trainingData: trainingData, textColumn: "text", labelColumn: "class") // According to the CSV file

let evaluationMetrics = sentimentClassifier.evaluation(on: testingData, textColumn: "text", labelColumn: "class")

let evaluatioonAccuracy = (1.0 - evaluationMetrics.classificationError)  * 100

let metadata = MLModelMetadata(author: "Lovice", shortDescription: "A Model trained to classify sentiment on Tweets", version: "1.00")

try sentimentClassifier.write(to: URL(fileURLWithPath: "/Users/lovice/Documents/Projects/TwitterSentiment/TweetSentimentClassifier.mlmodel"))

try sentimentClassifier.prediction(from: "@XYZ is a bad company!") // Negative Tweet

try sentimentClassifier.prediction(from: "I like @CocaCola is refreshing and tasty") // Neutral

try sentimentClassifier.prediction(from: "@F1 is the best motorsport") // Postive
