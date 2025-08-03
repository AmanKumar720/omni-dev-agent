# GitHub Setup Automation Script for Omni-Dev Agent
# This script prepares your project for GitHub sharing

Write-Host "🚀 Setting up Omni-Dev Agent for GitHub..." -ForegroundColor Green

# Check if we're in a git repository
if (-not (Test-Path ".git")) {
    Write-Host "❌ Not a git repository. Please run 'git init' first." -ForegroundColor Red
    exit 1
}

# Add all files to staging
Write-Host "📁 Adding all files to git..." -ForegroundColor Yellow
git add .

# Check if there are changes to commit
$status = git status --porcelain
if ($status) {
    Write-Host "💾 Committing latest changes..." -ForegroundColor Yellow
    git commit -m "Final project setup with documentation and examples

- Added comprehensive PROJECT_SHOWCASE.md
- Generated sample project plans (4 examples)
- Enhanced documentation with usage examples
- Ready for GitHub sharing"
} else {
    Write-Host "✅ No new changes to commit." -ForegroundColor Green
}

# Display current status
Write-Host "`n📊 Current Git Status:" -ForegroundColor Cyan
git log --oneline -5

Write-Host "`n📋 Files ready for sharing:" -ForegroundColor Cyan
Get-ChildItem -Filter "*.md" | ForEach-Object {
    Write-Host "  📄 $($_.Name) ($([math]::Round($_.Length/1KB, 1)) KB)" -ForegroundColor White
}

Write-Host "`n🎯 Next Steps:" -ForegroundColor Green
Write-Host "1. Open GitHub Desktop application" -ForegroundColor Yellow
Write-Host "2. Click 'Add an Existing Repository from Hard Drive'" -ForegroundColor Yellow
Write-Host "3. Browse to: $PWD" -ForegroundColor Yellow
Write-Host "4. Publish to GitHub (make it public for sharing)" -ForegroundColor Yellow
Write-Host "5. Copy the GitHub URL to share with others" -ForegroundColor Yellow

Write-Host "`n🔗 Or create repository manually:" -ForegroundColor Green
Write-Host "1. Go to https://github.com/new" -ForegroundColor Yellow
Write-Host "2. Repository name: omni-dev-agent" -ForegroundColor Yellow
Write-Host "3. Description: AI-powered development planning and project management tool" -ForegroundColor Yellow
Write-Host "4. Make it Public" -ForegroundColor Yellow
Write-Host "5. Don't initialize with README (we already have one)" -ForegroundColor Yellow

Write-Host "`n💡 After creating GitHub repository, run:" -ForegroundColor Green
Write-Host "git remote add origin https://github.com/YOUR-USERNAME/omni-dev-agent.git" -ForegroundColor Cyan
Write-Host "git branch -M main" -ForegroundColor Cyan  
Write-Host "git push -u origin main" -ForegroundColor Cyan

Write-Host "`n✨ Project is ready for sharing!" -ForegroundColor Green
