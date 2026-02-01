# 🚀 Python Backend Deployment Guide (Render)

## ✅ What We've Prepared

All necessary files have been created for deploying your Python indicator backend to Render:

### Files Created:
- ✅ `requirements.txt` - Python dependencies
- ✅ `build.sh` - Build script for Render
- ✅ `Procfile` - Process definition for Render
- ✅ `render.yaml` - Render service configuration
- ✅ `server.py` - Updated with production settings

### Code Updated:
- ✅ Frontend automatically detects production/development
- ✅ CORS configured for your live site
- ✅ Environment-based PORT configuration

---

## 📋 Deployment Steps

### Step 1: Push Backend Code to GitHub

**Option A: Separate Repository (Recommended)**
```bash
cd "C:\Users\Sethu\Downloads\Binance\Charting plaform\python-backend"

# Initialize git if not already done
git init

# Add all files
git add .

# Commit
git commit -m "Initial Python backend for custom indicators"

# Create a new repository on GitHub called "charting-indicators-backend"
# Then link it:
git remote add origin https://github.com/YOUR_USERNAME/charting-indicators-backend.git
git branch -M main
git push -u origin main
```

**Option B: Use Existing Repository**
```bash
# If your frontend is already in a git repo, just push the changes
cd "C:\Users\Sethu\Downloads\Binance\Charting plaform"
git add python-backend/
git commit -m "Add Python backend for custom indicators"
git push
```

---

### Step 2: Deploy to Render

1. **Go to Render Dashboard**
   - Visit: https://dashboard.render.com/
   - Sign in (or create account - it's free!)

2. **Create New Web Service**
   - Click **"New +"** → **"Web Service"**

3. **Connect GitHub Repository**
   - Select your repository (charting-indicators-backend or your main repo)
   - Click **"Connect"**

4. **Configure Service Settings**
   
   **Name:** `charting-indicators-backend`
   
   **Region:** Choose closest to your frontend (e.g., Oregon USA)
   
   **Branch:** `main`
   
   **Root Directory:** 
   - If Option A (separate repo): Leave blank
   - If Option B (monorepo): Enter `python-backend`
   
   **Runtime:** `Python 3`
   
   **Build Command:**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Start Command:**
   ```bash
   uvicorn server:app --host 0.0.0.0 --port $PORT
   ```
   
   **Instance Type:** `Free` (for now)

5. **Environment Variables (if needed)**
   - Usually not needed for basic setup
   - If you want to add custom variables, click "Advanced" and add them

6. **Click "Create Web Service"**
   - Render will start building and deploying
   - This takes about 3-5 minutes

---

### Step 3: Get Your Backend URL

Once deployed, Render will give you a URL like:
```
https://charting-indicators-backend.onrender.com
```

**Important:** The frontend code is already configured to use this exact URL in production! If Render gives you a different URL, you'll need to update these files:

- `src/services/indicatorService.ts` (line 8)
- `src/components/ScriptEditor/ScriptEditor.tsx` (line 39)
- `src/components/IndicatorPanel.tsx` (line 70)

Just replace `https://charting-indicators-backend.onrender.com` with your actual Render URL.

---

### Step 4: Verify Backend is Running

Visit your backend URL in a browser:
```
https://charting-indicators-backend.onrender.com/
```

You should see:
```json
{
  "status": "running",
  "service": "CryptoChart Pro Indicator Server"
}
```

---

### Step 5: Test Custom Indicators on Live Site

1. Visit your live site: https://charting-platform.onrender.com/

2. Click **"Indicators"** button

3. Click **"Custom (Python)"** category

4. Click **"Create New Indicator"**

5. The Python editor should open with a template

6. Click **"Save & Apply"**

7. If everything works, you should see "Server Online" ✅

8. The custom indicator will be added to your chart!

---

## 🔧 Troubleshooting

### Backend Shows "Offline" on Live Site

**Check:**
1. Is the Render service "Live" (green status)?
2. Does visiting the backend URL manually work?
3. Are there any CORS errors in browser console (F12)?

**Fix:**
- If CORS error, make sure `server.py` includes your frontend URL in allowed origins
- Check Render logs for errors: Dashboard → Your Service → Logs

### "Failed to fetch" Error

**Common Causes:**
1. Backend URL mismatch
2. Backend service is sleeping (free tier sleeps after 15 min of inactivity)
3. HTTPS mixed content (backend must be HTTPS if frontend isHTTPS)

**Fix:**
- Wait 30 seconds for Render to wake up the service
- Check browser Network tab (F12) to see the actual error

### Render Build Fails

**Check:**
- Is `requirements.txt` correct?
- Are all files committed to GitHub?
- Check Render build logs for the specific error

---

## 💰 Costs

- **Free Tier:** 
  - ✅ Perfect for testing
  - ⚠️ Service spins down after 15 minutes of inactivity
  - ⚠️ Takes ~30 seconds to wake up on first request
  - ✅ 750 hours/month free (enough for hobby use)

- **Paid Starter ($7/month):**
  - ✅ Always on (no spin-down)
  - ✅ Instant response
  - ✅ Better for production use

---

## 📦 What Custom Indicators Can Do

Once deployed, users can:

1. **Create Python Indicators** via the built-in editor
2. **Use Built-in TA Functions:** `ta.sma()`, `ta.ema()`, `ta.rsi()`, `ta.macd()`, etc.
3. **Save Custom Scripts** that persist server-side
4. **Test Indicators** with live chart data
5. **Share Scripts** (if you build that feature later)

Example Custom Indicator:
```python
indicator = {
    "name": "My SMA",
    "overlay": True,
    "inputs": {"length": 20}
}

def calculate(data, inputs):
    close = data["close"]
    sma = close.rolling(inputs["length"]).mean()
    return {"plot": sma}
```

---

## 🎯 Next Steps

After backend is deployed:

1. **Test thoroughly** on the live site
2. **Add more built-in indicators** to `python-backend/builtin/`
3. **Create example scripts** for users
4. **Add sharing/community features** (optional)
5. **Upgrade to paid tier** when you're ready for production

---

## 📞 Need Help?

If you encounter issues:
1. Check Render service logs
2. Check browser console (F12)
3. Test backend URL directly
4. Verify all files are pushed to GitHub

---

**Last Updated:** 2026-02-01  
**Status:** Ready to Deploy! 🚀
