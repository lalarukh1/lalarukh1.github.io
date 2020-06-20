class TweetCriteria:

    def __init__(self):
        self.maxTweets = 0
        self.topTweets = False

    def setSince(self, since):
        self.since = since
        return self

    def setUntil(self, until):
        self.until = until
        return self

    def setQuerySearch(self, querySearch):
        self.querySearch = querySearch
        return self

    def setLang(self, Lang):
        self.lang = Lang
        return self
