from nltk.sentiment import SentimentIntensityAnalyzer


class SentimentModel:

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def predict(self, text):

        score = self.analyzer.polarity_scores(text)["compound"]

        if score >= 0.05:
            sentiment = "Positive"
        elif score <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        return {
            "text": text,
            "score": score,
            "sentiment": sentiment
        }

    def overall_sentiment(self, headlines):

        if len(headlines) == 0:
            return {
                "average_score": 0,
                "overall_sentiment": "Neutral"
            }

        scores = []

        for text in headlines:
            scores.append(self.analyzer.polarity_scores(text)["compound"])

        avg = sum(scores) / len(scores)

        if avg >= 0.05:
            sentiment = "Positive"
        elif avg <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        return {
            "average_score": avg,
            "overall_sentiment": sentiment
        }
