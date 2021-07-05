def process_holiday(train_df, test_df):
    # 공휴일 전후
    train_df['공휴일전후'] = 0
    test_df['공휴일전후'] = 0
    train_df['공휴일전후'][17] = 1
    train_df['공휴일전후'][3] = 1
    train_df['공휴일전후'][62] = 1
    train_df['공휴일전후'][131] = 1
    train_df['공휴일전후'][152] = 1
    train_df['공휴일전후'][226] = 1
    train_df['공휴일전후'][221] = 1
    train_df['공휴일전후'][224] = 1
    train_df['공휴일전후'][245] = 1
    train_df['공휴일전후'][310] = 1  # 2
    train_df['공휴일전후'][311] = 1
    train_df['공휴일전후'][309] = 1
    train_df['공휴일전후'][330] = 1
    train_df['공휴일전후'][379] = 1
    train_df['공휴일전후'][467] = 1
    train_df['공휴일전후'][470] = 1
    train_df['공휴일전후'][502] = 1
    train_df['공휴일전후'][565] = 1
    train_df['공휴일전후'][623] = 1
    train_df['공휴일전후'][651] = 1
    train_df['공휴일전후'][705] = 1
    train_df['공휴일전후'][709] = 1
    train_df['공휴일전후'][815] = 1
    train_df['공휴일전후'][864] = 1
    train_df['공휴일전후'][950] = 1
    train_df['공휴일전후'][951] = 1
    train_df['공휴일전후'][953] = 1
    train_df['공휴일전후'][954] = 1
    train_df['공휴일전후'][955] = 1
    train_df['공휴일전후'][971] = 1  # 2
    train_df['공휴일전후'][1038] = 1
    train_df['공휴일전후'][1099] = 1
    train_df['공휴일전후'][1129] = 1
    train_df['공휴일전후'][1187] = 1
    test_df['공휴일전후'][10] = 1
    test_df['공휴일전후'][20] = 1

    train_df['공휴일길이'] = 0
    test_df['공휴일길이'] = 0
    train_df['공휴일길이'][17] = 1  # 삼일절 전
    train_df['공휴일길이'][6] = 2  # 설 연휴 전
    train_df['공휴일길이'][47] = 1  # 총선 전
    train_df['공휴일길이'][62] = 4  # 어린이 날 전
    train_df['공휴일길이'][82] = 4  # 현충일 연휴 전(금요일)
    train_df['공휴일길이'][131] = 3  # 광복절 연휴 전(금요일)
    train_df['공휴일길이'][152] = 5  # 한가위 연휴 전
    train_df['공휴일길이'][162] = 3  # 개천절 연휴 전(금요일)
    train_df['공휴일길이'][226] = 2  # 연말
    train_df['공휴일길이'][221] = 2  # 크리스마스 전(금요일)
    train_df['공휴일길이'][245] = 4  # 설 연휴 전
    train_df['공휴일길이'][310] = 3  # 어린이 날 연휴 전(금요일)
    train_df['공휴일길이'][309] = 1  # 석가탄신일 전
    train_df['공휴일길이'][330] = 1  # 현충일 전
    train_df['공휴일길이'][379] = 1  # 광복절 전
    train_df['공휴일길이'][412] = 11  # 추석연휴 전
    train_df['공휴일길이'][466] = 3  # 성탄절 연휴 전(금요일)
    train_df['공휴일길이'][470] = 3  # 연말
    train_df['공휴일길이'][502] = 4  # 설 연휴 전
    train_df['공휴일길이'][510] = 3  # 삼일절 전
    train_df['공휴일길이'][565] = 1  # 석가탄신일 전
    train_df['공휴일길이'][575] = 1  # 현충일 전
    train_df['공휴일길이'][623] = 1  # 광복절 전
    train_df['공휴일길이'][650] = 2  # 추석 연휴 전
    train_df['공휴일길이'][651] = 16  # 한글날 전
    train_df['공휴일길이'][705] = 1  # 크리스마스 이브
    train_df['공휴일길이'][709] = 1  # 연말
    train_df['공휴일길이'][732] = 5  # 설연휴 전
    train_df['공휴일길이'][748] = 3  # 삼일절 연휴 전
    train_df['공휴일길이'][792] = 3  # 어린이날 연휴 전
    train_df['공휴일길이'][814] = 1  # 현충일 전
    train_df['공휴일길이'][863] = 1  # 광복절 전
    train_df['공휴일길이'][882] = 4  # 추석 연휴 전
    train_df['공휴일길이'][894] = 1  # 개천절 전
    train_df['공휴일길이'][897] = 1  # 한글날 전
    train_df['공휴일길이'][951] = 1  # 크리스마스 전
    train_df['공휴일길이'][955] = 1  # 연말
    train_df['공휴일길이'][971] = 4  # 설 연휴
    train_df['공휴일길이'][1027] = 1  # 국회의원선거 전
    train_df['공휴일길이'][1037] = 4  # 석가탄신일 연휴 전
    train_df['공휴일길이'][1038] = 1  # 어린이날 전
    train_df['공휴일길이'][1129] = 7  # 추석연휴 전
    train_df['공휴일길이'][1133] = 3  # 한글날 연휴 전
    train_df['공휴일길이'][1187] = 10  # 성탄절 연휴 전

    test_df['공휴일길이'][10] = 4  # 설연휴전
    test_df['공휴일길이'][20] = 3  # 삼일절 연휴 전

    return train_df, test_df


def get_month_average(train):
    월1 = 0
    월2 = 0
    월3 = 0
    월4 = 0
    월5 = 0
    월6 = 0
    월7 = 0
    월8 = 0
    월9 = 0
    월10 = 0
    월11 = 0
    월12 = 0

    월1cnt = 0
    월2cnt = 0
    월3cnt = 0
    월4cnt = 0
    월5cnt = 0
    월6cnt = 0
    월7cnt = 0
    월8cnt = 0
    월9cnt = 0
    월10cnt = 0
    월11cnt = 0
    월12cnt = 0
    arrays = []
    for col in ['중식계', '석식계'] :
        for i in range(train.shape[0]) :
            if train.loc[i, 'month'] == 1 :
                월1 += train.loc[i, col]
                월1cnt += 1
            elif train.loc[i, 'month'] == 2 :
                월2 += train.loc[i, col]
                월2cnt += 1
            elif train.loc[i, 'month'] == 3 :
                월3 += train.loc[i, col]
                월3cnt += 1
            elif train.loc[i, 'month'] == 4 :
                월4 += train.loc[i, col]
                월4cnt += 1
            elif train.loc[i, 'month'] == 5 :
                월5 += train.loc[i, col]
                월5cnt += 1
            elif train.loc[i, 'month'] == 6 :
                월6 += train.loc[i, col]
                월6cnt += 1
            elif train.loc[i, 'month'] == 7 :
                월7 += train.loc[i, col]
                월7cnt += 1
            elif train.loc[i, 'month'] == 8 :
                월8 += train.loc[i, col]
                월8cnt += 1
            elif train.loc[i, 'month'] == 9 :
                월9 += train.loc[i, col]
                월9cnt += 1
            elif train.loc[i, 'month'] == 10 :
                월10 += train.loc[i, col]
                월10cnt += 1
            elif train.loc[i, 'month'] == 11 :
                월11 += train.loc[i, col]
                월11cnt += 1
            elif train.loc[i, 'month'] == 12 :
                월12 += train.loc[i, col]
                월12cnt += 1

        월1 /= 월1cnt
        월2 /= 월2cnt
        월3 /= 월3cnt
        월4 /= 월4cnt
        월5 /= 월5cnt
        월6 /= 월6cnt
        월7 /= 월7cnt
        월8 /= 월8cnt
        월9 /= 월9cnt
        월10 /= 월10cnt
        월11 /= 월11cnt
        월12 /= 월12cnt

        array = [월1, 월2, 월3, 월4, 월5, 월6, 월7, 월8, 월9, 월10, 월11, 월12]
        arrays.append(array)

    return arrays