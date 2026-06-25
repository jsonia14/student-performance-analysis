## Student Performance Analysis (Flask)

### What this app does
- Loads `students.csv` with 60 students in alphabetical order and columns:
  - `reg_no`, `student_name`, `study_time`, `attendance`, `previous_scores`, `sleep_hours`, `final_score`
- Cleans/preprocesses the data (numeric coercion, drop missing/dupes, basic bounds)
- Runs EDA and saves 2 plots into `static/`
  - `static/heatmap.png` (correlation heatmap)
  - `static/score_distribution.png` (final score distribution)
- Trains a model (best of Random Forest vs Linear Regression), saves it to `model.pkl`, and loads it on future runs
- Provides a web UI to predict `final_score` from 4 inputs and shows improvement suggestions
- Also captures `student_name` and `reg_no`, then displays attendance, previous marks, predicted marks, and a performance category (`Bad`, `Average`, `Good`, `Excellent`)
- Adds a staff portal backed by SQLite:
  - Staff login/logout
  - Dashboard
  - Attendance app (per date)
  - Marks update app (update `previous_scores`)

### Project structure
- `app.py` (Flask backend + ML/EDA)
- `templates/index.html` (UI)
- `templates/login.html` (staff login)
- `templates/dashboard.html` (staff dashboard)
- `templates/attendance.html` (attendance app)
- `templates/marks.html` (marks update app)
- `static/style.css` (styles)
- `requirements.txt` (dependencies)

### How to run (Windows / VS Code)
1. Open this folder in VS Code.
2. Create and activate a virtual environment:

```bash
py -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```bash
py -m pip install -r requirements.txt
```

4. Run the app:

```bash
py app.py
```

5. Open `http://127.0.0.1:5000` in your browser.

### Staff login (default)
- **URL**: `http://127.0.0.1:5000/staff/login`
- **Username**: `admin`
- **Password**: `admin123`

After login you can open:
- `http://127.0.0.1:5000/dashboard`
- `http://127.0.0.1:5000/attendance`
- `http://127.0.0.1:5000/marks`

### Dataset notes
- If `students.csv` is missing, the app auto-generates a sample dataset so you can run immediately.
- To use your real dataset, replace `students.csv` with your file (same column names).

