import yfinance as yf


class NewsFetcher:

    def get_headlines(self, ticker="MSFT", limit=5):

        try:
            stock = yf.Ticker(ticker)
            news = stock.news

            headlines = []
            seen = set()

            for article in news:

                title = article.get("title")

                if not title:
                    continue

                # remove duplicates
                if title in seen:
                    continue

                seen.add(title)
                headlines.append(title)

                if len(headlines) >= limit:
                    break

            return headlines

        except Exception as e:
            print("News fetch error:", e)
            return []
