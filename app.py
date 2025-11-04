# app.py (일부 수정)

from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import timedelta
import os

app = Flask(__name__)

# --- 1. 기본 설정 ---
app.secret_key = os.urandom(24)
app.permanent_session_lifetime = timedelta(minutes=30) 

# --- 2. ⭐️ 데이터베이스 설정 (PostgreSQL로 변경) ⭐️ ---

# PostgreSQL 연결 문자열 형식:
# "postgresql://[유저이름]:[비밀번호]@[호스트주소]:[포트]/[DB이름]"

# ⭐️ 아래 정보를 본인의 pgAdmin4 접속 정보로 수정하세요 ⭐️
DB_USER = "postgres"  # (예: 'postgres')
DB_PASSWORD = "postgres" # (pgAdmin4 로그인 시 설정한 비번)
DB_HOST = "localhost" # (보통 'localhost' 또는 '127.0.0.1')
DB_PORT = "5432"      # (PostgreSQL 기본 포트)
DB_NAME = "music_db"  # (pgAdmin4에서 미리 생성할 데이터베이스 이름)

app.config['SQLALCHEMY_DATABASE_URI'] = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- 3. 데이터베이스 모델 (테이블) 정의 ---
# (이전 코드와 동일 - 변경 없음)
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# --- 4. 라우팅 (URL 연결) ---
# (이전 코드와 동일 - 변경 없음)
@app.route('/')
@app.route('/login', methods=['GET', 'POST'])
def login():
    # ... (이전 코드와 동일)
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['logged_in'] = True
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('index'))
        else:
            flash('잘못된 사용자 이름 또는 비밀번호입니다.')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # 1. 폼 데이터 가져오기
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        password_confirm = request.form['password-confirm']

        # 2. 비밀번호 일치 확인
        if password != password_confirm:
            flash('비밀번호가 일치하지 않습니다. 다시 확인해주세요.')
            return redirect(url_for('register'))

        # 3. 중복 확인
        existing_user = User.query.filter_by(username=username).first()
        existing_email = User.query.filter_by(email=email).first()
        
        if existing_user:
            flash('이미 존재하는 사용자 이름입니다.')
            return redirect(url_for('register'))
        
        if existing_email:
            flash('이미 사용 중인 이메일입니다.')
            return redirect(url_for('register'))
        
        # ⭐️ 4. DB에 저장 (올바른 위치)
        # 이 코드들이 'return redirect' 전에 와야 합니다.
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        
        # 5. 성공 메시지 및 리디렉션
        flash('회원가입에 성공했습니다! 로그인해주세요.')
        return redirect(url_for('login'))
    
    # GET 요청일 경우 (처음 페이지 열 때)
    return render_template('register.html')

@app.route('/index')
def index():
    # ... (이전 코드와 동일)
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('index.html', username=session.get('username'))

@app.route('/logout')
def logout():
    # ... (이전 코드와 동일)
    session.pop('logged_in', None)
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect(url_for('login'))

# --- 5. Flask 앱 실행 및 DB 생성 ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all() # ⭐️ DB_NAME에 해당하는 DB에 'user' 테이블을 생성합니다.
    app.run(debug=True)