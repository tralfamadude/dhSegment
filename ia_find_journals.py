from internetarchive import get_session
from internetarchive import download
from datetime import datetime

class FindJournals:
    def __init__(self, publication_id: str):
        self.publication_id = publication_id

    def iso8601_to_datetime(self, date_string) -> datetime:
        # '2008-01-01T00:00:00Z'
        date_string = date_string.replace("Z", "+00:00")
        date = datetime.fromisoformat(date_string)
        return date

    def find(self, begin_datetime=None, end_datetime=None) -> list:
        """
        search for issues identifiers. If time range not specified, then all issues found will be returned.
        :param begin_datetime: find content at or after this datetime.
        :param end_datetime: find content before this time.
        :return:
        """
        results = []
        sess = get_session()
        sess.mount_http_adapter()
        if begin_datetime is not None:
            begin_datetime = self.iso8601_to_datetime(begin_datetime)
        if end_datetime is not None:
            end_datetime = self.iso8601_to_datetime(end_datetime)
        sresults = sess.search_items(f"collection:{self.publication_id}", fields=['date'])
        for item in sresults:
            if begin_datetime is None or end_datetime is None:
                results.append(item['identifier'])
            else:
                ds = item['date']
                dtime = self.iso8601_to_datetime(ds)
                if dtime >= begin_datetime and dtime < end_datetime:
                    results.append(item['identifier'])
        return results

if __name__ == '__main__':
    publication_id = 'pub_journal-of-thought'
    find_journals = FindJournals(publication_id)
    issues = find_journals()


