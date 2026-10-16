# Running the Project

This guide covers running both the backend and frontend locally

## 1. Backend

Open a PowerShell terminal and run:

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host localhost --port 8000 --reload
```

Once running, the backend is available at:
- Documentation: http://localhost:8000/docs
- Health check: http://localhost:8000/api/health

## 2. Frontend

Keep the backend running. Open a second PowerShell terminal and run:

```powershell
cd frontend
npm run dev
```

Once running, open the application in your browser:
- Application home: http://localhost:3000
- Audio analysis: http://localhost:3000/analyze
- Live voice control: http://localhost:3000/live
- Evaluation dashboard: http://localhost:3000/dashboard
