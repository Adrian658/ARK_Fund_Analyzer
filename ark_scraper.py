import pandas as pd
import requests
import functools
import logging
import PyPDF2
import os
import io

from pymongo import MongoClient
import boto3
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail


EXTERNAL_PATH = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(EXTERNAL_PATH, "ark.env")

MDB_SERVERNAME = "InvestmentDB"
MDB_DBNAME = "ARK_fund"

# Construct URLs where ARK holdings files are located
WEBSITE_URL = "https://ark-funds.com/wp-content/fundsiteliterature/"
construct_url = lambda suffix, ftype: os.path.join(WEBSITE_URL, suffix, ftype)

ARK_ETF_LIST = [
    "ARK_INNOVATION_ETF_ARKK_HOLDINGS", 
    "ARK_AUTONOMOUS_TECHNOLOGY_&_ROBOTICS_ETF_ARKQ_HOLDINGS",
    "ARK_NEXT_GENERATION_INTERNET_ETF_ARKW_HOLDINGS",
    "ARK_GENOMIC_REVOLUTION_MULTISECTOR_ETF_ARKG_HOLDINGS",
    "ARK_FINTECH_INNOVATION_ETF_ARKF_HOLDINGS"
]

URLS_PDF = [construct_url("holdings", "%s.%s" % (x, "pdf")) for x in ARK_ETF_LIST]
URLS_CSV = [construct_url("csv", "%s.%s" % (x, "csv")) for x in ARK_ETF_LIST]

# Logging config
logger = logging.getLogger()
logfile = logging.FileHandler(filename=os.path.join(EXTERNAL_PATH, 'ark.log'), mode='a')
logformatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logfile.setFormatter(logformatter)
logger.addHandler(logfile)

# Tracks if any errors occur during runtime
ERRORS = []




def extract_from_pdf(file_stream):
    """
    Extract the fund name and date from pdf byte stream

    Parameters
    ----------
    file_stream : byte stream of pdf file
    """


    ark_pdf = PyPDF2.PdfFileReader(file_stream)
    
    pageObj = ark_pdf.getPage(0)
    pageText = pageObj.extractText()

    splitText = pageText.split("\n")
    portfolio = splitText[0].split(' ')[-2].replace('(', '').replace(')', '')
    date = splitText[1].split(' ')[2].replace('/', '.')
    
    return portfolio, date


def create_mdb_conn(server):
    """
    Creates and returns connection to MongoDB Atlas server instance.
    Credentials are read from env variables

    Parameters
    ----------
    server : the name of the MDB server to establish connection with
    """

    # Fetch credentials
    username = os.environ['MONGO_ATLAS_USERNAME']
    password = os.environ['MONGO_ATLAS_PASSWORD']

    # Create connection to MDB and returnthe connection
    client = MongoClient("mongodb+srv://%s:%s@%s.ccy3i.mongodb.net/%s?retryWrites=true&w=majority" % (username, password, server.lower(), server))
    return client



def scrape_ark_csv(url):
    """
    Request holdings csv from ARK fund website and return it

    Parameters
    ----------
    url : url where the csv file is located at
    """

    # Send request to ARKK website to get document with details of current days holdings
    resp = requests.get(url)

    # Call helper function to parse the csv file
    data = list(map(split_ark_holding_record, resp.text.split('\n')))
    headers = data[0]
    del data[0]

    # Convert parsed data into a dataframe and format fields as necessary
    # Data is of the form: 
    #   company, date, fund, ticker, cusip, shares, market value($), weight(%)
    df = pd.DataFrame(data, columns=headers).dropna()
    df.columns = df.columns.str.strip('"')
    df['company'] = df['company'].str.strip('"')
    df['ticker'] = df['ticker'].str.strip('"')
    df['date'] = pd.to_datetime(df['date'])

    # Reorder columns
    df = df[['ticker', 'company', 'date', 'fund', 'cusip', 'shares', 'market value($)', 'weight(%)']]

    return df

def extract_from_ark_holdings(df):
    """
    Extract the date and fund name from the holdings data and ensure that all dates/funds in the data are the same

    Parameters
    ----------
    df : dataframe containing ARK holdings
    """
        
    unq_dates = df['date'].unique()
    unq_funds = df['fund'].unique()
    if len(unq_dates) == len(unq_funds) == 1:
        date = str(unq_dates[0]).replace('-', '.')
        portfolio = unq_funds[0]
    else:
        err_msg = "%s: Unique dates or funds > 1" % url
        raise ValueError(err_msg)

    return portfolio, date

def split_ark_holding_record(row):
    """
    Parse a single row from ARK Holdings csv

    Parameters
    ----------
    row : a single row fron ARK Holdings csv file 
    """

    row = row.split(',')
    # If length indicates a valid holding record and is not empty
    if len(row) == 8 and not functools.reduce(lambda x, y: (x and y) == '', row):
        return row
    return []

def send_email_alert(errmessages, from_addr="aguzman658@gmail.com", to_addr="aguzman658@gmail.com"):
    """
    Sends an email to myself with the specified error messages
    Used for alerting of errors in runtime

    Parameters
    ----------
    errmessages : list of error messages to include in email body
    """

    apikey = os.environ['SENGRID_API_KEY']
    
    email = Mail(
        from_email=from_addr,
        to_emails=to_addr,
        subject="ERROR requesting today's ARKK Holdings",
        html_content=
        """
        <h2>An error occured while trying to update today's holdings from the ARKK website</h2>
        <h5>A list of known errors is below, but please check the logs for more detailed information</h5>
        <ol>%s</ol>
        <p>Thank you precious</p>
        """ % "".join(["<li><b>%s</b></li>" % x for x in errmessages])
    )

    sg = SendGridAPIClient(apikey)
    response = sg.send(email)
    if response.status_code not in [200, 202]:
        raise ValueError("Received %s response: %s" % (response.status_code, response.body))





if __name__ == "__main__":

    # Ensure environment variables are created
    os.system("source ./%s" % ENV_FILE)

    # Connect to MDB Atlas where ARK data is stored
    try:
        conn = create_mdb_conn(MDB_SERVERNAME)
        db = conn[MDB_DBNAME]

        s3 = boto3.client('s3')
    except Exception as e:
        err_msg = "Error connecting to MongoDB Atlas or S3"
        logger.critical("\n".join([err_msg, str(e)]))
        send_email_alert([err_msg, e])
        exit(1)

    # Iterate through csv/pdf urls
    for csv, pdf in zip(URLS_CSV, URLS_PDF):

        # Store csv file holding data in MongoDB
        try:
            # Request csv file from csv url
            holdings = scrape_ark_csv(csv)

            # Ensure the date and fund from the csv are valid
            fund, date = extract_from_ark_holdings(holdings)
        except Exception as e:
            logger.error(e)
            ERRORS.append(e)
        else:
            # Write holdings to MDB Atlas
            try:
                collection = db["%s_Holdings" % fund]
                collection.insert_many(holdings.to_dict('records'))
            except Exception as e:
                err_msg = "Failed write to MongoDB Atlas: %s" % csv
                logger.critical("\n".join([err_msg, str(e)]))
                ERRORS.extend([err_msg, e])

        # Store pdf holding files in Amazon S3
        try:
            response = requests.get(url)
            if response.status_code != 200:
                pass

            file_stream = io.BytesIO(response.content)
            portfolio, date = extract_from_pdf(file_stream)

            s3_key = "%s_Holdings_%s" % (portfolio, date)

            s3.upload_fileobj(file_stream, bucket, s3_key)
        except Exception as e:
            err_msg = "Failed write to AWS S3: %s" % pdf
            logger.critical("\n".join([err_msg, str(e)]))
            ERRORS.extend([err_msg, e])

    # Close connection to MongoDB Atlas
    conn.close()

    # If any errors occured send email with list of errors and prompt to check logs
    if ERRORS:
        try:
            send_email_alert(ERRORS)
        except Exception as e:
            logger.critical(e)
