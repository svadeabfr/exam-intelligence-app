"""
Exam Intelligence App — Integrated Single File
Combines backend logic + Streamlit frontend
"""

# ============================================================
# IMPORTS
# ============================================================
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from openai import OpenAI

# ============================================================
# DATABASE
# ============================================================

question_bank = {
    ("1st", "Semester 1"): {
        "Maths": [
            "Solve: ∫x dx",
            "Explain Pythagoras theorem"
        ],
        "Physics": [
            "State Newton's 2nd law",
            "Explain Bernoulli's principle"
        ]
    },
    ("1st", "Semester 2"): {
        "Computer Science": [
            "Explain OOP concepts",
            "What is time complexity of binary search?"
        ],
        "Maths": [
            "Find eigenvalues of a 2x2 matrix"
        ]
    },
    ("2nd", "Semester 3"): {
        "Physics": [
            "Derive equation for kinetic energy",
            "Explain Bernoulli's principle with example"
        ]
    }
    # ➡️ Add more year/semester mappings here
}

users = {
    "student1":  {"password": "123", "role": "Student"},
    "teacher1":  {"password": "123", "role": "Teacher"},
    "examcell1": {"password": "123", "role": "Exam Cell"}
}

# ============================================================
# OPENAI CLIENT
# ============================================================

client = OpenAI(api_key="YOUR_OPENAI_API_KEY")   # 🔴 Replace with your key

# ============================================================
# ML MODEL  (trained once at module load)
# ============================================================

_train_questions = [
    "What is 2+2?",
    "Explain Newton's laws",
    "Derive Schrödinger equation",
    "Define variable in Python",
    "Explain OOP concepts",
    "Analyze time complexity of quicksort"
]
_labels = [0, 1, 2, 0, 1, 2]   # 0=Easy, 1=Medium, 2=Hard

_vectorizer = TfidfVectorizer()
_X = _vectorizer.fit_transform(_train_questions)

_model = LogisticRegression()
_model.fit(_X, _labels)

# ============================================================
# BACKEND FUNCTIONS
# ============================================================

def get_question_bank(year, semester):
    return question_bank.get((year, semester), {})


def detect_repetition(new_question, old_questions):
    """Returns max cosine-similarity between new_question and old_questions."""
    if not old_questions:
        return 0.0
    vec = TfidfVectorizer()
    all_q = old_questions + [new_question]
    tfidf = vec.fit_transform(all_q)
    similarity = cosine_similarity(tfidf[-1], tfidf[:-1])
    return float(similarity.max())


def predict_difficulty(question):
    """Returns 'Easy', 'Medium', or 'Hard'."""
    vec = _vectorizer.transform([question])
    pred = _model.predict(vec)[0]
    return ["Easy", "Medium", "Hard"][pred]


def generate_ai_questions(subject, difficulty, old_questions, num_questions=2):
    """Calls OpenAI to generate non-repetitive exam questions."""
    prompt = (
        f"Generate {num_questions} exam questions for subject: {subject}\n"
        f"Difficulty: {difficulty}\n\n"
        "Rules:\n"
        "- Exam oriented\n"
        "- No repetition\n"
        "- Output ONLY bullet points (one question per line)"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.choices[0].message.content
        questions = [
            q.strip("- ").strip()
            for q in text.split("\n")
            if q.strip()
        ]
        final = [
            q for q in questions
            if detect_repetition(q, old_questions) < 0.7
        ]
        return final[:num_questions]
    except Exception as e:
        st.warning(f"AI generation skipped: {e}")
        return []


def generate_question_paper(year, semester, difficulty):
    """
    Builds a question paper string for the given year/semester/difficulty.
    Mixes existing bank questions with AI-generated ones.
    """
    subjects = get_question_bank(year, semester)
    if not subjects:
        return None

    paper = f"QUESTION PAPER — {year} Year | {semester} | Difficulty: {difficulty}\n"
    paper += "=" * 60 + "\n"

    for subject, questions in subjects.items():
        paper += f"\n--- {subject} ---\n"

        # Existing questions filtered by difficulty
        selected = random.sample(questions, min(2, len(questions)))
        for q in selected:
            level = predict_difficulty(q)
            if difficulty == level or difficulty == "Medium":
                paper += f"  • {q}  [{level}]\n"

        # AI-generated questions
        ai_qs = generate_ai_questions(subject, difficulty, questions, num_questions=2)
        for q in ai_qs:
            paper += f"  • {q}  [AI – {difficulty}]\n"

    return paper


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Exam Intelligence App",
    page_icon="🎓",
    layout="wide"
)

# ============================================================
# SESSION STATE INIT
# ============================================================

for key in ["role", "year", "semester", "logged_in"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ============================================================
# LOGIN PAGE
# ============================================================

if not st.session_state.get("logged_in"):
    st.title("🎓 Exam Intelligence App")
    st.subheader("🔐 Login")

    col1, col2 = st.columns([1, 1])
    with col1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

    with col2:
        year_choice     = st.selectbox("Select Year", ["1st", "2nd", "3rd", "4th"])
        semester_choice = st.selectbox("Select Semester", [
            "Semester 1", "Semester 2", "Semester 3", "Semester 4",
            "Semester 5", "Semester 6", "Semester 7", "Semester 8"
        ])

    if st.button("Login"):
        if username in users and users[username]["password"] == password:
            st.session_state["role"]      = users[username]["role"]
            st.session_state["logged_in"] = True
            st.session_state["year"]      = year_choice
            st.session_state["semester"]  = semester_choice
            st.success(f"Welcome, {st.session_state['role']}!")
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.stop()   # Prevent rendering dashboards before login

# ============================================================
# SIDEBAR (post-login)
# ============================================================

role     = st.session_state["role"]
year     = st.session_state["year"]
semester = st.session_state["semester"]

st.sidebar.header(f"👤 {role}")
st.sidebar.write(f"📅 {year} Year — {semester}")
page = st.sidebar.radio("Navigate", ["Dashboard", "Logout"])

if page == "Logout":
    for key in ["role", "year", "semester", "logged_in"]:
        st.session_state[key] = None
    st.rerun()

# ============================================================
# ── STUDENT DASHBOARD ──────────────────────────────────────
# ============================================================

if role == "Student":
    st.header("📚 Student Dashboard")

    # ── Practice PYQs ────────────────────────────────────────
    st.subheader("🏋️ Practice PYQs")
    st.button("Start Practice", key="student_practice")

    st.divider()

    # ── Question Bank ────────────────────────────────────────
    st.subheader("📂 Question Bank")

    if (year, semester) in question_bank:
        subjects       = question_bank[(year, semester)]
        subject_choice = st.selectbox("Select Subject", list(subjects.keys()), key="qb_subject")

        if st.button("View Question Bank", key="qb_view"):
            st.success(f"Questions for {subject_choice} — {year}, {semester}:")
            for i, q in enumerate(subjects[subject_choice], 1):
                st.write(f"{i}. {q}")

            st.download_button(
                label="📥 Download QB (TXT)",
                data="\n".join(subjects[subject_choice]),
                file_name=f"{year}_{semester}_{subject_choice}_QB.txt",
                mime="text/plain",
                key="student_qb_download"
            )
    else:
        st.warning("No Question Bank available for this Year & Semester yet.")

    st.divider()

    # ── Question Paper Generator ─────────────────────────────
    st.subheader("📝 Question Paper Generator")
    difficulty = st.selectbox("Select Difficulty", ["Easy", "Medium", "Hard"], key="student_difficulty")

    if st.button("Generate Paper", key="student_generate"):
        with st.spinner("Generating paper…"):
            paper = generate_question_paper(year, semester, difficulty)

        if paper:
            st.success("Paper generated successfully!")
            st.text(paper)

            st.download_button(
                label="📥 Download Paper (TXT)",
                data=paper,
                file_name=f"Student_{year}_{semester}_{difficulty}_paper.txt",
                mime="text/plain",
                key="student_paper_txt"
            )
            st.download_button(
                label="📥 Download Paper (Word)",
                data=paper,
                file_name=f"Student_{year}_{semester}_{difficulty}_paper.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="student_paper_docx"
            )
        else:
            st.warning("No questions found for this Year & Semester combination.")

    st.divider()

    # ── Formula Sheet ─────────────────────────────────────────
    st.subheader("📖 Formula Sheet")
    with st.expander("View Formula Sheet"):
        st.markdown("**Algebra**")
        st.latex(r"(a+b)^2 = a^2 + 2ab + b^2")
        st.latex(r"(a-b)^2 = a^2 - 2ab + b^2")
        st.latex(r"a^2 - b^2 = (a-b)(a+b)")

        st.markdown("**Geometry**")
        st.latex(r"\text{Area of Circle} = \pi r^2")
        st.latex(r"\text{Circumference} = 2\pi r")
        st.latex(r"\text{Area of Triangle} = \tfrac{1}{2} b h")

        st.markdown("**Trigonometry**")
        st.latex(r"\sin^2\theta + \cos^2\theta = 1")
        st.latex(r"\tan\theta = \tfrac{\sin\theta}{\cos\theta}")
        st.latex(r"\cot\theta = \tfrac{1}{\tan\theta}")

        st.markdown("**Calculus**")
        st.latex(r"\frac{d}{dx}(x^n) = nx^{n-1}")
        st.latex(r"\int x^n\,dx = \frac{x^{n+1}}{n+1} + C")

        st.markdown("**Physics**")
        st.latex(r"F = ma")
        st.latex(r"V = IR")
        st.latex(r"E = mc^2")
        st.latex(r"KE = \tfrac{1}{2}mv^2")
        st.latex(r"PE = mgh")

    st.divider()

    # ── Flashcards ────────────────────────────────────────────
    st.subheader("🃏 Flashcards")
    with st.expander("View Flashcards"):
        flashcards = {
            "Expand (a+b)²":                "a² + 2ab + b²",
            "Factorize a² − b²":            "(a−b)(a+b)",
            "Area of a circle?":            "πr²",
            "Area of a triangle?":          "½ × base × height",
            "sin²θ + cos²θ = ?":            "1",
            "Define tanθ":                  "sinθ / cosθ",
            "Derivative of xⁿ?":            "n·xⁿ⁻¹",
            "Integral of xⁿ?":              "(xⁿ⁺¹)/(n+1) + C",
            "Newton's Second Law?":         "F = ma",
            "Formula for kinetic energy?":  "½mv²",
            "Formula for potential energy?": "mgh",
        }
        for question, answer in flashcards.items():
            with st.expander(f"Q: {question}"):
                st.write(f"A: {answer}")

    st.divider()

    # ── Progress Tracking ─────────────────────────────────────
    st.subheader("📊 Progress Tracking")
    st.line_chart([10, 20, 15, 30])


# ============================================================
# ── TEACHER DASHBOARD ──────────────────────────────────────
# ============================================================

elif role == "Teacher":
    st.header("👩‍🏫 Teacher Dashboard")

    # ── Create Assignment ─────────────────────────────────────
    st.subheader("📋 Create Assignment")
    uploaded_assignment = st.file_uploader("Upload Assignment", key="teacher_upload")
    if st.button("Assign to Students", key="teacher_assign"):
        if uploaded_assignment:
            st.success("Assignment created and assigned!")
        else:
            st.warning("Please upload a file first.")

    st.divider()

    # ── Student Analytics ─────────────────────────────────────
    st.subheader("📈 Student Analytics")
    st.line_chart([5, 15, 25, 35])

    st.divider()

    # ── Question Paper Generator ─────────────────────────────
    st.subheader("📝 Question Paper Generator")
    difficulty = st.selectbox("Select Difficulty", ["Easy", "Medium", "Hard"], key="teacher_difficulty")

    if st.button("Generate Paper", key="teacher_generate"):
        with st.spinner("Generating paper…"):
            paper = generate_question_paper(year, semester, difficulty)

        if paper:
            st.success("Paper generated successfully!")
            st.text(paper)

            st.download_button(
                label="📥 Download Paper (TXT)",
                data=paper,
                file_name=f"Teacher_{year}_{semester}_{difficulty}_paper.txt",
                mime="text/plain",
                key="teacher_paper_txt"
            )
            st.download_button(
                label="📥 Download Paper (Word)",
                data=paper,
                file_name=f"Teacher_{year}_{semester}_{difficulty}_paper.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="teacher_paper_docx"
            )
        else:
            st.warning("No questions found for this Year & Semester combination.")

    st.divider()

    # ── Case Study Questions ──────────────────────────────────
    st.subheader("🔬 Case Study Questions")
    if st.button("Generate Case Study Questions", key="teacher_case_study"):
        st.info("Case study questions generated using AI (demo).")

    st.divider()

    # ── Assignment Push ───────────────────────────────────────
    st.subheader("📤 Push Assignment to Classes")
    uploaded_push = st.file_uploader("Upload Assignment for Distribution", key="teacher_push_upload")
    if st.button("Push to Classes", key="teacher_push"):
        if uploaded_push:
            st.success("Assignment pushed to all classes!")
        else:
            st.warning("Please upload a file first.")


# ============================================================
# ── EXAM CELL DASHBOARD ────────────────────────────────────
# ============================================================

elif role == "Exam Cell":
    st.header("🏢 Exam Cell Dashboard")

    # ── Centralized Question Bank ─────────────────────────────
    st.subheader("📂 Centralized Question Bank")

    if (year, semester) in question_bank:
        subjects       = question_bank[(year, semester)]
        subject_choice = st.selectbox("Select Subject", list(subjects.keys()), key="ec_qb_subject")

        if st.button("View Question Bank", key="ec_qb_view"):
            st.success(f"Questions for {subject_choice} — {year}, {semester}:")
            for i, q in enumerate(subjects[subject_choice], 1):
                st.write(f"{i}. {q}")

            st.download_button(
                label="📥 Download QB (TXT)",
                data="\n".join(subjects[subject_choice]),
                file_name=f"{year}_{semester}_{subject_choice}_QB.txt",
                mime="text/plain",
                key="ec_qb_download"
            )
    else:
        st.warning("No Question Bank available for this Year & Semester yet.")

    st.divider()

    # ── Question Paper Generator ─────────────────────────────
    st.subheader("📝 Question Paper Generator")
    difficulty = st.selectbox("Select Difficulty", ["Easy", "Medium", "Hard"], key="ec_difficulty")

    if st.button("Generate Paper", key="ec_generate"):
        with st.spinner("Generating paper…"):
            paper = generate_question_paper(year, semester, difficulty)

        if paper:
            st.success("Paper generated successfully!")
            st.text(paper)

            st.download_button(
                label="📥 Download Paper (TXT)",
                data=paper,
                file_name=f"ExamCell_{year}_{semester}_{difficulty}_paper.txt",
                mime="text/plain",
                key="ec_paper_txt"
            )
            st.download_button(
                label="📥 Download Paper (Word)",
                data=paper,
                file_name=f"ExamCell_{year}_{semester}_{difficulty}_paper.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="ec_paper_docx"
            )
        else:
            st.warning("No questions found for this Year & Semester combination.")

    st.divider()

    # ── Analytics Dashboard ───────────────────────────────────
    st.subheader("📊 Analytics Dashboard")
    st.line_chart([12, 18, 22, 28])

    st.divider()

    # ── Exam Model Paper Upload ───────────────────────────────
    st.subheader("📄 Exam Model Paper")
    uploaded_pattern = st.file_uploader(
        "Upload paper pattern (mass distribution)", key="ec_model_upload"
    )
    if st.button("Publish to Classes", key="ec_publish"):
        if uploaded_pattern:
            st.success("Exam model paper published to all classes!")
        else:
            st.warning("Please upload a file first.")