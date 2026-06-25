import os
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from flask import Flask, render_template, request
from matplotlib import pyplot as plt
from flask_login import (
    LoginManager,
    UserMixin,
    login_required,
    login_user,
    logout_user,
    current_user,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from werkzeug.security import check_password_hash, generate_password_hash


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "students.csv"
MODEL_PATH = APP_DIR / "model.pkl"
STATIC_DIR = APP_DIR / "static"
DB_PATH = APP_DIR / "school.db"

FEATURES = ["study_time", "attendance", "previous_scores", "sleep_hours"]
TARGET = "final_score"
ALL_COLUMNS = FEATURES + [TARGET]
STUDENT_FIELDS = ["student_name", "reg_no"]


def _ensure_dirs() -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    (APP_DIR / "templates").mkdir(parents=True, exist_ok=True)


def _generate_sample_dataset(path: Path, n: int = 250, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)

    study_time = rng.uniform(0.5, 6.0, size=n)  # hours/day
    attendance = rng.uniform(50, 100, size=n)  # percent
    previous_scores = rng.uniform(35, 95, size=n)  # previous average
    sleep_hours = rng.uniform(4.5, 9.0, size=n)  # hours/night

    noise = rng.normal(0, 6.0, size=n)
    final_score = (
        8.0 * study_time
        + 0.25 * attendance
        + 0.55 * previous_scores
        + 2.5 * sleep_hours
        + noise
    )
    final_score = np.clip(final_score, 0, 100)

    df = pd.DataFrame(
        {
            "study_time": np.round(study_time, 2),
            "attendance": np.round(attendance, 2),
            "previous_scores": np.round(previous_scores, 2),
            "sleep_hours": np.round(sleep_hours, 2),
            "final_score": np.round(final_score, 2),
        }
    )
    df.to_csv(path, index=False)


def _generate_60_students_dataset(path: Path, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)

    first_names = [
        "Aarav",
        "Aditi",
        "Aisha",
        "Akash",
        "Aman",
        "Ananya",
        "Aniket",
        "Anjali",
        "Ansh",
        "Arjun",
        "Arya",
        "Avni",
        "Bhavya",
        "Chaitanya",
        "Diya",
        "Eesha",
        "Farhan",
        "Gauri",
        "Harsh",
        "Isha",
        "Jatin",
        "Kavya",
        "Kiran",
        "Laksh",
        "Meera",
        "Mihir",
        "Naina",
        "Nikhil",
        "Om",
        "Pari",
        "Pranav",
        "Priya",
        "Rahul",
        "Rhea",
        "Ritvik",
        "Riya",
        "Saanvi",
        "Sahil",
        "Samir",
        "Sara",
        "Shaurya",
        "Shreya",
        "Siddharth",
        "Simran",
        "Tanvi",
        "Tanya",
        "Tejas",
        "Trisha",
        "Uday",
        "Vaibhav",
        "Varun",
        "Ved",
        "Vihaan",
        "Vikram",
        "Yash",
        "Zara",
        "Aarohi",
        "Bharat",
        "Devansh",
        "Ishaan",
    ]
    last_names = [
        "Agarwal",
        "Bansal",
        "Chopra",
        "Desai",
        "Gupta",
        "Iyer",
        "Jain",
        "Kapoor",
        "Khan",
        "Mehta",
        "Nair",
        "Patel",
        "Reddy",
        "Sharma",
        "Singh",
        "Verma",
    ]

    names = []
    for i in range(60):
        fn = first_names[i % len(first_names)]
        ln = last_names[(i * 3) % len(last_names)]
        names.append(f"{fn} {ln}")
    names = sorted(names, key=lambda s: s.lower())

    reg_nos = [f"REG2026{str(i + 1).zfill(3)}" for i in range(60)]

    study_time = rng.uniform(0.5, 6.0, size=60)
    attendance = rng.uniform(55, 100, size=60)
    previous_scores = rng.uniform(35, 95, size=60)
    sleep_hours = rng.uniform(4.5, 9.0, size=60)

    noise = rng.normal(0, 6.0, size=60)
    final_score = (
        8.0 * study_time
        + 0.25 * attendance
        + 0.55 * previous_scores
        + 2.5 * sleep_hours
        + noise
    )
    final_score = np.clip(final_score, 0, 100)

    df = pd.DataFrame(
        {
            "reg_no": reg_nos,
            "student_name": names,
            "study_time": np.round(study_time, 2),
            "attendance": np.round(attendance, 2),
            "previous_scores": np.round(previous_scores, 2),
            "sleep_hours": np.round(sleep_hours, 2),
            "final_score": np.round(final_score, 2),
        }
    )
    df.to_csv(path, index=False)


def load_and_clean_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        _generate_60_students_dataset(path)

    df = pd.read_csv(path)

    # Normalize column names and keep only expected columns
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Expected: {ALL_COLUMNS}. Found: {list(df.columns)}"
        )
    # allow extra columns like reg_no, student_name
    df = df.copy()

    # Coerce to numeric and handle missing values
    for c in ALL_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=ALL_COLUMNS).drop_duplicates()

    # Basic sanity bounds
    df["attendance"] = df["attendance"].clip(0, 100)
    df["previous_scores"] = df["previous_scores"].clip(0, 100)
    df["final_score"] = df["final_score"].clip(0, 100)
    df["study_time"] = df["study_time"].clip(lower=0)
    df["sleep_hours"] = df["sleep_hours"].clip(lower=0)

    if len(df) < 20:
        raise ValueError("Not enough clean rows to train a model (need at least 20).")

    return df


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = db_connect()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS staff_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS students (
            reg_no TEXT PRIMARY KEY,
            student_name TEXT NOT NULL,
            study_time REAL NOT NULL,
            attendance REAL NOT NULL,
            previous_scores REAL NOT NULL,
            sleep_hours REAL NOT NULL,
            final_score REAL NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS attendance_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reg_no TEXT NOT NULL,
            date TEXT NOT NULL,
            present INTEGER NOT NULL,
            UNIQUE(reg_no, date),
            FOREIGN KEY(reg_no) REFERENCES students(reg_no)
        )
        """
    )

    # Create default staff user if none exists
    cur.execute("SELECT COUNT(*) AS c FROM staff_users")
    if int(cur.fetchone()["c"]) == 0:
        cur.execute(
            "INSERT INTO staff_users (username, password_hash) VALUES (?, ?)",
            ("admin", generate_password_hash("admin123")),
        )

    conn.commit()
    conn.close()


def seed_students_from_csv(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    required = ["reg_no", "student_name"] + ALL_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        # if the CSV is numeric-only (older version), regenerate 60-student dataset
        _generate_60_students_dataset(csv_path)
        df = pd.read_csv(csv_path)

    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM students")
    if int(cur.fetchone()["c"]) > 0:
        conn.close()
        return

    df = df.copy()
    df["reg_no"] = df["reg_no"].astype(str)
    df["student_name"] = df["student_name"].astype(str)
    for c in ALL_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=ALL_COLUMNS)

    rows = df[
        ["reg_no", "student_name", "study_time", "attendance", "previous_scores", "sleep_hours", "final_score"]
    ].to_records(index=False)
    cur.executemany(
        """
        INSERT OR REPLACE INTO students
        (reg_no, student_name, study_time, attendance, previous_scores, sleep_hours, final_score)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [tuple(r) for r in rows],
    )
    conn.commit()
    conn.close()


def export_students_to_csv(csv_path: Path) -> None:
    conn = db_connect()
    df = pd.read_sql_query(
        "SELECT reg_no, student_name, study_time, attendance, previous_scores, sleep_hours, final_score FROM students ORDER BY student_name",
        conn,
    )
    conn.close()
    df.to_csv(csv_path, index=False)


def create_eda_plots(df: pd.DataFrame) -> dict:
    sns.set_theme(style="whitegrid")
    paths: dict[str, str] = {}

    # Heatmap of correlations
    plt.figure(figsize=(7.5, 5.5))
    corr = df[ALL_COLUMNS].corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", square=True, cbar=True)
    plt.title("Correlation Heatmap")
    heatmap_path = STATIC_DIR / "heatmap.png"
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=160)
    plt.close()
    paths["heatmap"] = f"static/{heatmap_path.name}"

    # Distribution of final scores
    plt.figure(figsize=(7.5, 5.0))
    sns.histplot(df[TARGET], kde=True, bins=18, color="#2f6f8e")
    plt.title("Final Score Distribution")
    plt.xlabel("Final Score")
    plt.ylabel("Count")
    dist_path = STATIC_DIR / "score_distribution.png"
    plt.tight_layout()
    plt.savefig(dist_path, dpi=160)
    plt.close()
    paths["distribution"] = f"static/{dist_path.name}"

    return paths


def train_or_load_model(df: pd.DataFrame) -> tuple[object, dict]:
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Try Random Forest first (often stronger on mixed relationships); fall back to Linear
    rf = RandomForestRegressor(
        n_estimators=250, random_state=42, n_jobs=-1, min_samples_leaf=2
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_metrics = {
        "model_type": "RandomForestRegressor",
        "r2": float(r2_score(y_test, rf_pred)),
        "mae": float(mean_absolute_error(y_test, rf_pred)),
    }

    lin = Pipeline(
        steps=[("scaler", StandardScaler()), ("model", LinearRegression())]
    )
    lin.fit(X_train, y_train)
    lin_pred = lin.predict(X_test)
    lin_metrics = {
        "model_type": "LinearRegression",
        "r2": float(r2_score(y_test, lin_pred)),
        "mae": float(mean_absolute_error(y_test, lin_pred)),
    }

    best_model, best_metrics = (rf, rf_metrics) if rf_metrics["r2"] >= lin_metrics["r2"] else (lin, lin_metrics)

    payload = {
        "model": best_model,
        "metrics": best_metrics,
        "feature_order": FEATURES,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)

    return best_model, best_metrics


def load_model() -> tuple[object, dict] | tuple[None, dict]:
    if not MODEL_PATH.exists():
        return None, {}
    try:
        with open(MODEL_PATH, "rb") as f:
            payload = pickle.load(f)
        return payload.get("model"), payload.get("metrics", {})
    except Exception:
        return None, {}


def suggestions_from_inputs(df: pd.DataFrame, x: dict) -> list[str]:
    tips: list[str] = []

    means = df[FEATURES].mean(numeric_only=True)
    # Simple, interpretable advice based on common thresholds + dataset averages.
    if x["study_time"] < max(2.0, float(means["study_time"]) * 0.9):
        tips.append("Increase study time (aim for at least 2 hours/day).")
    if x["attendance"] < max(75.0, float(means["attendance"]) * 0.9):
        tips.append("Improve attendance (target 75%+ consistently).")
    if x["previous_scores"] < max(60.0, float(means["previous_scores"]) * 0.9):
        tips.append("Review fundamentals to boost previous scores.")
    if x["sleep_hours"] < 7.0:
        tips.append("Try to get 7–9 hours of sleep for better focus.")

    if not tips:
        tips.append("Great balance — keep your study routine and consistency.")

    return tips


def performance_category(score: float) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Average"
    return "Bad"


def _to_float(value: str) -> float:
    return float(str(value).strip())


_ensure_dirs()
init_db()

# Ensure the CSV is the requested 60-student dataset and seed DB on first run.
if not DATA_PATH.exists():
    _generate_60_students_dataset(DATA_PATH)
else:
    # if older CSV without reg_no/name, upgrade it
    try:
        tmp = pd.read_csv(DATA_PATH, nrows=1)
        if "reg_no" not in tmp.columns or "student_name" not in tmp.columns:
            _generate_60_students_dataset(DATA_PATH)
    except Exception:
        _generate_60_students_dataset(DATA_PATH)

seed_students_from_csv(DATA_PATH)
export_students_to_csv(DATA_PATH)

df_global = load_and_clean_data(DATA_PATH)
plot_paths = create_eda_plots(df_global)

model, metrics = load_model()
if model is None:
    model, metrics = train_or_load_model(df_global)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")

login_manager = LoginManager()
login_manager.login_view = "staff_login"
login_manager.init_app(app)


class StaffUser(UserMixin):
    def __init__(self, user_id: int, username: str):
        self.id = str(user_id)
        self.username = username


@login_manager.user_loader
def load_user(user_id: str):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT id, username FROM staff_users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return StaffUser(row["id"], row["username"])


@app.get("/")
def index():
    return render_template(
        "index.html",
        prediction=None,
        category=None,
        student_info=None,
        tips=[],
        error=None,
        plot_paths=plot_paths,
        metrics=metrics,
        form_values={k: "" for k in FEATURES + STUDENT_FIELDS},
        current_user=current_user,
    )


@app.route("/staff/login", methods=["GET", "POST"])
def staff_login():
    error = None
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        conn = db_connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, username, password_hash FROM staff_users WHERE username = ?",
            (username,),
        )
        row = cur.fetchone()
        conn.close()

        if row and check_password_hash(row["password_hash"], password):
            login_user(StaffUser(row["id"], row["username"]))
            return render_template(
                "dashboard.html",
                current_user=current_user,
                stats=get_dashboard_stats(),
            )
        error = "Invalid username or password."

    return render_template("login.html", error=error, current_user=current_user)


@app.route("/staff/signup", methods=["GET", "POST"])
def staff_signup():
    if current_user and current_user.is_authenticated:
        return render_template(
            "dashboard.html",
            current_user=current_user,
            stats=get_dashboard_stats(),
        )

    error = None
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        confirm = request.form.get("confirm_password") or ""

        if len(username) < 3:
            error = "Username must be at least 3 characters."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        elif password != confirm:
            error = "Passwords do not match."
        else:
            conn = db_connect()
            cur = conn.cursor()
            try:
                cur.execute(
                    "INSERT INTO staff_users (username, password_hash) VALUES (?, ?)",
                    (username, generate_password_hash(password)),
                )
                conn.commit()
                user_id = cur.lastrowid
                conn.close()
                login_user(StaffUser(user_id, username))
                return render_template(
                    "dashboard.html",
                    current_user=current_user,
                    stats=get_dashboard_stats(),
                )
            except sqlite3.IntegrityError:
                conn.close()
                error = "Username already exists. Please choose another."

    return render_template("signup.html", error=error, current_user=current_user)


@app.get("/staff/logout")
@login_required
def staff_logout():
    logout_user()
    return render_template(
        "login.html",
        error=None,
        current_user=current_user,
        message="Logged out successfully.",
    )


def get_dashboard_stats() -> dict:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM students")
    students_count = int(cur.fetchone()["c"])
    cur.execute("SELECT COUNT(*) AS c FROM attendance_records")
    attendance_count = int(cur.fetchone()["c"])
    conn.close()
    return {
        "students_count": students_count,
        "attendance_count": attendance_count,
        "model_type": (metrics or {}).get("model_type"),
        "r2": (metrics or {}).get("r2"),
        "mae": (metrics or {}).get("mae"),
    }


@app.get("/dashboard")
@login_required
def dashboard():
    return render_template(
        "dashboard.html",
        current_user=current_user,
        stats=get_dashboard_stats(),
    )


@app.route("/attendance", methods=["GET", "POST"])
@login_required
def attendance():
    date = (request.values.get("date") or pd.Timestamp.today().strftime("%Y-%m-%d")).strip()
    saved = False

    conn = db_connect()
    cur = conn.cursor()
    students = cur.execute(
        "SELECT reg_no, student_name FROM students ORDER BY student_name"
    ).fetchall()

    if request.method == "POST":
        for s in students:
            key = f"present_{s['reg_no']}"
            present = 1 if request.form.get(key) == "on" else 0
            cur.execute(
                """
                INSERT INTO attendance_records (reg_no, date, present)
                VALUES (?, ?, ?)
                ON CONFLICT(reg_no, date) DO UPDATE SET present = excluded.present
                """,
                (s["reg_no"], date, present),
            )
        conn.commit()
        saved = True

    existing = {
        row["reg_no"]: int(row["present"])
        for row in cur.execute(
            "SELECT reg_no, present FROM attendance_records WHERE date = ?",
            (date,),
        ).fetchall()
    }
    conn.close()

    return render_template(
        "attendance.html",
        current_user=current_user,
        students=students,
        date=date,
        existing=existing,
        saved=saved,
    )


@app.route("/marks", methods=["GET", "POST"])
@login_required
def marks():
    saved = False
    conn = db_connect()
    cur = conn.cursor()
    students = cur.execute(
        "SELECT reg_no, student_name, previous_scores FROM students ORDER BY student_name"
    ).fetchall()

    if request.method == "POST":
        for s in students:
            key = f"prev_{s['reg_no']}"
            raw = request.form.get(key, "")
            try:
                val = float(raw)
            except Exception:
                continue
            val = float(np.clip(val, 0, 100))
            cur.execute(
                "UPDATE students SET previous_scores = ? WHERE reg_no = ?",
                (val, s["reg_no"]),
            )
        conn.commit()
        saved = True
        export_students_to_csv(DATA_PATH)

    conn.close()

    # Refresh global df/plots when marks updated (so EDA uses latest data)
    global df_global, plot_paths
    df_global = load_and_clean_data(DATA_PATH)
    plot_paths = create_eda_plots(df_global)

    return render_template(
        "marks.html",
        current_user=current_user,
        students=students,
        saved=saved,
    )


@app.post("/predict")
def predict():
    form_values = {k: request.form.get(k, "") for k in FEATURES + STUDENT_FIELDS}
    try:
        student_name = form_values["student_name"].strip()
        reg_no = form_values["reg_no"].strip()
        if not student_name:
            raise ValueError("Student name is required.")
        if not reg_no:
            raise ValueError("Registration number is required.")

        x = {
            "study_time": _to_float(form_values["study_time"]),
            "attendance": _to_float(form_values["attendance"]),
            "previous_scores": _to_float(form_values["previous_scores"]),
            "sleep_hours": _to_float(form_values["sleep_hours"]),
        }

        # Gentle bounds to reduce weird inputs
        x["study_time"] = float(np.clip(x["study_time"], 0, 24))
        x["attendance"] = float(np.clip(x["attendance"], 0, 100))
        x["previous_scores"] = float(np.clip(x["previous_scores"], 0, 100))
        x["sleep_hours"] = float(np.clip(x["sleep_hours"], 0, 24))

        X = pd.DataFrame([[x[f] for f in FEATURES]], columns=FEATURES)
        pred = float(model.predict(X)[0])
        pred = float(np.clip(pred, 0, 100))
        category = performance_category(pred)

        tips = suggestions_from_inputs(df_global, x)
        return render_template(
            "index.html",
            prediction=round(pred, 2),
            category=category,
            student_info={
                "student_name": student_name,
                "reg_no": reg_no,
                "attendance": round(x["attendance"], 2),
                "previous_scores": round(x["previous_scores"], 2),
            },
            tips=tips,
            error=None,
            plot_paths=plot_paths,
            metrics=metrics,
            form_values=form_values,
            current_user=current_user,
        )
    except Exception as e:
        return render_template(
            "index.html",
            prediction=None,
            category=None,
            student_info=None,
            tips=[],
            error=str(e),
            plot_paths=plot_paths,
            metrics=metrics,
            form_values=form_values,
            current_user=current_user,
        )


if __name__ == "__main__":
    # Use PORT env var if provided (helpful on some setups)
    port = int(os.environ.get("PORT", "5000"))
    app.run(debug=True, host="127.0.0.1", port=port)

