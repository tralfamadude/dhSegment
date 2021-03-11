from internetarchive import get_session
from internetarchive import download
from datetime import datetime
import argparse
import sys


class FindJournals:
    """
    This simplifies finding journal issue IDs.
    """

    def __init__(self, publication_id: str):
        self.publication_id = publication_id

    def iso8601_to_datetime(self, date_string) -> datetime:
        # '2008-01-01T00:00:00Z'
        date = datetime.datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%SZ')
        return date

    def find(self, begin_datetime: str = None, end_datetime: str = None) -> list:
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
                ds = item['date']  # example: "2010-01-01T00:00:00Z"
                dtime = self.iso8601_to_datetime(ds)
                if dtime >= begin_datetime and dtime < end_datetime:
                    results.append(item['identifier'])
        return results


if __name__ == '__main__':
    # Example usage:
    # publication_id = 'pub_journal-of-thought'
    # find_journals = FindJournals(publication_id)
    # issues = find_journals.find()

    FLAGS = None
    # init the parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--journal', '-j',
        type=str,
        help='journal id, example: pub_journal-of-thought'
    )
    parser.add_argument(
        '--after', '-a',
        type=str,
        default=None,
        help='optional iso date for picking issues at or after, example: 1970-01-31'
    )
    parser.add_argument(
        '--before', '-b',
        type=str,
        default=None,
        help='optional iso date for picking issues before, example: 2010-01-31'
    )
    FLAGS, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        print(f"  Unknown args: {unparsed}")
        sys.exit(1)
    journal_id = FLAGS.journal
    after = FLAGS.after
    before = FLAGS.before
    if journal_id is None:
        parser.print_help()
        sys.exit(1)
    find_journals = FindJournals(journal_id)
    issues = find_journals.find(after, before)
    for issue_id in issues:
        print(issue_id)
    sys.exit(0)

