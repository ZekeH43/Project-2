# Transaction data

## File used by the experiment

`Resources/credit card data.csv` is the CSV already committed in this repository. The cleanup does not change its contents.

| Property | Value |
| --- | --- |
| Rows | 1,000,000 |
| Non-fraud (`fraud=0`) | 912,597 |
| Fraud (`fraud=1`) | 87,403 (8.7403%) |
| Features | 7 |
| Missing values | 0 in the committed copy |
| Exact duplicate rows | 0 in the committed copy |
| SHA-256 | `9bb7f58c9297ca61a4ded0dba65c9cb8cd1b1313542024151491590489909d62` |

The original repository did not record the publisher, download URL, dataset version, license, or collection method. Those details remain unverified. No real-world performance or privacy claims should be inferred from the column names or high model scores. Confirm the original source before making claims about data ownership or redistribution rights; this cleanup does not assign a new license to the existing files.

## Schema

These descriptions follow the column names used in the original analysis. Distance units and the exact definitions of transaction history are not recorded in the repository.

| Column | Meaning | Validation |
| --- | --- | --- |
| `distance_from_home` | Distance from the cardholder's home | Finite, nonnegative number |
| `distance_from_last_transaction` | Distance from the preceding transaction | Finite, nonnegative number |
| `ratio_to_median_purchase_price` | Purchase amount relative to a median purchase amount | Finite, nonnegative number |
| `repeat_retailer` | Indicator for a repeated retailer | 0 or 1 |
| `used_chip` | Indicator for chip use | 0 or 1 |
| `used_pin_number` | Indicator for PIN use | 0 or 1 |
| `online_order` | Indicator for an online order | 0 or 1 |
| `fraud` | Target label: 1 fraud, 0 not fraud | 0 or 1; excluded from model inputs |

Custom files must contain exactly these columns, in any order. At least 20 rows and two examples of each class are required; the chosen sample and split must also leave both classes represented in train and test. Validation rejects missing values, unexpected columns, invalid flags, and duplicate rows. It does not silently repair or delete data.

Rejecting identical rows prevents exact copies from appearing across the random split. It does not rule out related transactions or customer-level leakage: customer identifiers are not available for that check.

## Historical assets

- `Resources/credit_card_data_2.csv`: the original smaller 499-row file, with 458 non-fraud and 41 fraud rows. It has a UTF-8 byte-order mark, which the loader supports. Its smaller sample gives less stable evaluation results.
- `Resources/credit_card_data_2.xlsx`: the historical spreadsheet export; not needed by the code.
- `FinalPP.pptx`, `Presentation.pptx`, and `Presentation (2).pptx`: original presentation versions. Their metrics and conclusions predate the corrections.
- `heatmap.png`: an original plot file. The current notebook generates its own training-data plots.

Notebook checkpoints and the empty `data_collection.py` placeholder were removed. The checkpoint CSV was byte-for-byte identical to the main CSV. This removes a duplicate from the current checkout; it does not shrink the existing Git history.
