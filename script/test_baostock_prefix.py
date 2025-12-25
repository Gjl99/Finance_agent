import baostock as bs
import pandas as pd

def test_baostock_code(code):
    lg = bs.login()
    print(f"Login respond error_code: {lg.error_code}")
    
    print(f"Testing code: {code}")
    rs = bs.query_stock_basic(code=code)
    print(f"query_stock_basic respond error_code: {rs.error_code}")
    print(f"query_stock_basic respond error_msg: {rs.error_msg}")
    
    data_list = []
    while rs.next():
        data_list.append(rs.get_row_data())
    
    print(f"Result count: {len(data_list)}")
    if data_list:
        print(data_list)
        
    bs.logout()

if __name__ == "__main__":
    print("--- Testing with prefix ---")
    test_baostock_code("sh.603871")
    print("\n--- Testing without prefix ---")
    test_baostock_code("603871")
