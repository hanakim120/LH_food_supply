import re

import numpy as np
import pandas as pd

from dateutil import rrule
from datetime import datetime

from selenium import webdriver


def get_weather():
    temps = []
    rains = []

    try:
        for year in range(2016, 2022) :
            for month in range(1, 13) :
                if year == 2021 and month == 7 :
                    break
                url = '''https://www.weather.go.kr/weather/climate/past_cal.jsp?stn=108&yy={}&mm={}&obs=1&x=25&y=14'''.format(
                    year, month)

                options = webdriver.ChromeOptions()
                options.add_argument("user-agent=Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko")

                driver = webdriver.Chrome(options=options)

                driver.get(url)
                driver.set_window_size(3000, 3000)
                content = driver.find_element_by_xpath('//*[@id="content_weather"]/table/tbody').text

                temps += re.findall('평균기온:[0-9-.]*', content)
                rains += re.findall('강수량:[0-9.]*', content)

                driver.quit()
    except:
        print('Execution Failed, Check your Chrome driver version')

    return temps, rains

def main(temps, rains):
    new_temps = np.array([float(t.replace('평균기온:', '')) for t in temps if t != '평균기온:'])
    new_rains = np.array([float(r.replace('강수량:', '')) if r != '강수량:' else 0.0 for r in rains[:1987]])

    a = '20160101'
    b = '20210609'

    result = []
    for dt in rrule.rrule(rrule.DAILY, dtstart=datetime.strptime(a, '%Y%m%d'), until=datetime.strptime(b, '%Y%m%d')) :
        result.append(dt.strftime('%Y%m%d'))

    new_result = pd.DataFrame(result, columns=['일자'])
    new_result.일자 = new_result.일자.apply(lambda x : x[:4] + '-' + x[4 :6] + '-' + x[6 :])

    df = pd.DataFrame(np.concatenate([new_temps.reshape(-1, 1), new_rains.reshape(-1, 1)], axis=1),
                      columns=['temp', 'rain'])
    df = pd.concat([new_result, df], axis=1)
    df.to_csv('temp.csv', index=False)

if __name__ == '__main__':
    temps, rains = get_weather()
    main(temps, rains)