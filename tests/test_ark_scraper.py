import unittest
import ark_scraper

import pandas as pd
from pymongo import MongoClient

STATIC_FILE_PATH = 'tests/static/'

class ARK_Test(unittest.TestCase):

    def test_extract_from_pdf(self):
        with open('%sARKK_10.16.2020.pdf' % STATIC_FILE_PATH, 'rb') as file:
            portfolio, date = ark_scraper.extract_from_pdf(file)
            self.assertEqual(portfolio, "ARKK", "Should be ARKK")
            self.assertEqual(date, "10.16.2020", "Should be 10.16.2020")

    def test_create_mdb_conn(self):
        conn = ark_scraper.create_mdb_conn(ark_scraper.MDB_SERVERNAME)
        self.assertIsInstance(conn,  MongoClient)
        conn.close()

    def test_scrape_ark_csv(self):
        url = "https://ark-funds.com/wp-content/fundsiteliterature/csv/ARK_INNOVATION_ETF_ARKK_HOLDINGS.csv"
        df = ark_scraper.scrape_ark_csv(url)
        self.assertEqual(list(df.columns), ['ticker', 'company', 'date', 'fund', 'cusip', 'shares', 'market value($)', 'weight(%)'])

    def test_extract_from_ark_holdings(self):
        df = pd.read_csv('%sARKK_12.04.2020.csv' % STATIC_FILE_PATH)
        fund, date = ark_scraper.extract_from_ark_holdings(df)
        self.assertEqual(fund, 'ARKK', "Should be ARKK")
        self.assertEqual(date, "2020.12.04", "Should be 2020.12.04")