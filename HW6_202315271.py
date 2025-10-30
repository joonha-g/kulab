from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
import time

# 크롬 옵션 설정 (다운로드 경로 지정 가능)
chrome_options = webdriver.ChromeOptions()
chrome_options.add_experimental_option("detach", True)  # 실행 후 창 유지
# chrome_options.add_experimental_option("prefs", {"download.default_directory": r"C:\Users\사용자\Downloads"})

# 크롬 드라이버 실행
driver = webdriver.Chrome(options=chrome_options)
driver.implicitly_wait(10)  # 최대 10초 대기

# 1. KPX 홈페이지 접속
driver.get("https://new.kpx.or.kr/main/")
time.sleep(20)
#닫기 버튼 클릭
try:
    close_btn = driver.find_element(By.XPATH, "//a[@class='control close']")
    close_btn.click()
    time.sleep(2)
except NoSuchElementException:
    print("닫기 버튼이 없음 → 넘어감")

# 2. 상단 메뉴 - 주요산업 클릭
driver.find_element(By.XPATH, "//a[@href='/menu.es?mid=a10401010000']").click()
time.sleep(3)

# 3. 우측 메뉴 - 전력관련정보 클릭
driver.find_element(By.LINK_TEXT, "전력관련정보").click()
time.sleep(2)

# 4. 계통한계가격(SMP) 클릭
driver.find_element(By.LINK_TEXT, "계통한계가격(SMP)").click()
time.sleep(3)

# 날짜 입력 박스 요소 가져오기
date_box = driver.find_element(By.ID, "issue_date")
time.sleep(2)
driver.execute_script("""
const el = document.getElementById('issue_date');
el.value = '2017-12-20';
el.dispatchEvent(new Event('input',  { bubbles: true }));
el.dispatchEvent(new Event('change', { bubbles: true }));
""")

# 검색 버튼 클릭
search_btn = driver.find_element(By.XPATH, "//input[@value='검색']")
search_btn.click()
time.sleep(2)


# 엑셀 다운로드 버튼 클릭
download_btn = driver.find_element(By.XPATH, "//a[contains(text(),'엑셀')]")  # '엑셀 다운로드' 버튼
download_btn.click()
time.sleep(5)  # 다운로드 대기

print("✅ 크롤링 완료: 엑셀 파일이 다운로드되었습니다.")

