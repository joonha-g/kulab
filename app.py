# app.py

from flask import Flask, render_template, request, redirect, url_for, session
from datetime import timedelta
import os

app = Flask(__name__)

# 세션 관리를 위한 SECRET_KEY 설정 (매우 중요! 실제 배포 시에는 복잡하게 변경)
app.secret_key = os.urandom(24) # 무작위 24바이트 키 생성
app.permanent_session_lifetime = timedelta(minutes=30) # 세션 유지 시간 설정

# --- 라우팅 (URL 연결) ---

# 루트 주소 (http://127.0.0.1:5000/)로 접속했을 때 메인 페이지를 보여줍니다.
@app.route('/')
def index():
    return render_template('index.html') # 이제 index.html을 렌더링합니다!


# 로그인 페이지 (기존과 동일)
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == 'test' and password == '1234': # 임시 로그인 성공 조건
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('main_page')) # 로그인 성공 시 메인 페이지로 리디렉션
        else:
            error = '잘못된 사용자 이름 또는 비밀번호입니다.'
            return render_template('login.html', error=error)
    
    return render_template('login.html')

# 로그인 성공 시 이동할 임시 메인 페이지
@app.route('/main_page')
def main_page():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return f"환영합니다, {session['username']}님! 로그인에 성공했습니다."


# --- Flask 앱 실행 ---
if __name__ == '__main__':
    app.run(debug=True)