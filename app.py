

from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
import joblib, json, os, sqlite3
from datetime import datetime
import numpy as np

app = Flask(__name__)
app.secret_key = 'cardiocheck-secret-key-2024'  # Change in production!

# ── Load ML model ──
model  = joblib.load('model/heart_model.pkl')
scaler = joblib.load('model/scaler.pkl')
with open('model/meta.json') as f:
    meta = json.load(f)

FEATURES = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']

# ── Database ──
def get_db():
    db = sqlite3.connect('users.db')
    db.row_factory = sqlite3.Row
    return db

def init_db():
    db = get_db()
    db.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    db.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            prediction INTEGER,
            probability REAL,
            inputs TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    db.commit()
    db.close()

init_db()

# ── Auth helper ──
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated

# ── Routes ──

@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return redirect(url_for('index'))

@app.route('/login')
def login_page():
    if 'user_id' in session:
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', user_name=session.get('user_name'))

@app.route('/app')
@login_required
def index():
    return render_template('index.html', meta=meta, user_name=session.get('user_name'))

# ── Auth ──
@app.route('/auth/signup', methods=['POST'])
def signup():
    data     = request.json
    name     = data.get('name','').strip()
    email    = data.get('email','').strip().lower()
    password = data.get('password','')

    if not name or not email or not password:
        return jsonify({'error': 'All fields are required'}), 400
    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400

    db = get_db()
    if db.execute('SELECT id FROM users WHERE email=?', (email,)).fetchone():
        db.close()
        return jsonify({'error': 'Email already registered. Please login.'}), 409

    hashed = generate_password_hash(password)
    db.execute('INSERT INTO users (name, email, password) VALUES (?,?,?)', (name, email, hashed))
    db.commit()
    user = db.execute('SELECT * FROM users WHERE email=?', (email,)).fetchone()
    session['user_id']   = user['id']
    session['user_name'] = user['name']
    db.close()
    return jsonify({'success': True, 'name': name})

@app.route('/auth/login', methods=['POST'])
def login():
    data     = request.json
    email    = data.get('email','').strip().lower()
    password = data.get('password','')

    db   = get_db()
    user = db.execute('SELECT * FROM users WHERE email=?', (email,)).fetchone()
    db.close()

    if not user or not check_password_hash(user['password'], password):
        return jsonify({'error': 'Invalid email or password'}), 401

    session['user_id']   = user['id']
    session['user_name'] = user['name']
    return jsonify({'success': True, 'name': user['name']})

@app.route('/auth/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))

# ── Predict ──
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    data     = request.json
    features = [float(data[f]) for f in FEATURES]
    arr      = np.array([features])

    if meta['best_model'] == 'Logistic Regression':
        arr_input = scaler.transform(arr)
    else:
        arr_input = arr

    prediction  = int(model.predict(arr_input)[0])
    probability = float(model.predict_proba(arr_input)[0][1])

    db = get_db()
    db.execute(
        'INSERT INTO predictions (user_id, prediction, probability, inputs) VALUES (?,?,?,?)',
        (session['user_id'], prediction, round(probability*100,1), json.dumps(data))
    )
    db.commit()
    db.close()

    return jsonify({
        'prediction':  prediction,
        'probability': round(probability*100, 1),
        'label': 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease'
    })

# ── User info (extended) ──
@app.route('/me')
@login_required
def me():
    user_id = session['user_id']
    db = get_db()

    user = db.execute(
        'SELECT * FROM users WHERE id=?',
        (user_id,)
    ).fetchone()

    count = db.execute(
        'SELECT COUNT(*) as c FROM predictions WHERE user_id=?',
        (user_id,)
    ).fetchone()['c']

    db.close()

    response = jsonify({
        'name': user['name'],
        'email': user['email'],
        'total_checks': count,
        'joined': user['created_at'],
        'created_at': user['created_at'],
        'user_id': user_id
    })

    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'

    return response
# ── History (extended with limit param) ──
@app.route('/history')
@login_required
def history():
    user_id = session['user_id']   # always read from session, never from request
    limit   = request.args.get('limit', 10, type=int)
    db      = get_db()
    rows    = db.execute(
        'SELECT * FROM predictions WHERE user_id=? ORDER BY created_at DESC LIMIT ?',
        (user_id, limit)
    ).fetchall()
    db.close()

    result = []
    for r in rows:
        inp = json.loads(r['inputs'])
        result.append({
            'id':         r['id'],
            'prediction': r['prediction'],
            'probability':r['probability'],
            'date':       r['created_at'],
            'age':        inp.get('age'),
        })

    response = jsonify(result)
    # Prevent browser from caching history — critical for multi-user correctness
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma']        = 'no-cache'
    response.headers['Expires']       = '0'
    return response

if __name__ == '__main__':
    app.run(debug=True)