# Simple Automated GitHub Setup
Write-Host "🚀 Setting up GitHub Repository..." -ForegroundColor Green

# First, let's restart PowerShell session to recognize GitHub CLI
Write-Host "📋 Please restart PowerShell/Terminal and run this command:" -ForegroundColor Yellow
Write-Host "refreshenv" -ForegroundColor Cyan

# Check if we can use GitHub CLI
Write-Host "`n🔍 Testing GitHub CLI..." -ForegroundColor Yellow

# Commit current changes
Write-Host "📁 Committing current changes..." -ForegroundColor Yellow
git add .
git commit -m "Automated GitHub setup - Ready for publishing"

Write-Host "`n🎯 Manual GitHub CLI Commands:" -ForegroundColor Green
Write-Host "1. Authenticate: gh auth login --web" -ForegroundColor Cyan
Write-Host "2. Create repo: gh repo create omni-dev-agent --public --source . --push" -ForegroundColor Cyan
Write-Host "3. View repo: gh repo view --web" -ForegroundColor Cyan

Write-Host "`n✨ After running these commands, your repository will be live!" -ForegroundColor Green
