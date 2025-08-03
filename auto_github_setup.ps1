# Automated GitHub Repository Setup Script
# This script will create and publish your repository to GitHub automatically

Write-Host "🚀 Automated GitHub Setup for Omni-Dev Agent" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green

# Check if GitHub CLI is available
try {
    $ghVersion = gh --version
    Write-Host "✅ GitHub CLI found: $($ghVersion[0])" -ForegroundColor Green
} catch {
    Write-Host "❌ GitHub CLI not found. Please restart PowerShell and try again." -ForegroundColor Red
    exit 1
}

# Check if user is authenticated with GitHub
Write-Host "`n🔐 Checking GitHub authentication..." -ForegroundColor Yellow
try {
    $user = gh auth status 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Already authenticated with GitHub" -ForegroundColor Green
    } else {
        Write-Host "🔐 Please authenticate with GitHub..." -ForegroundColor Yellow
        gh auth login --web
    }
} catch {
    Write-Host "🔐 Please authenticate with GitHub..." -ForegroundColor Yellow
    gh auth login --web
}

# Prepare repository
Write-Host "`n📁 Preparing repository..." -ForegroundColor Yellow
git add .
git commit -m "Final automated setup - Ready for GitHub" --allow-empty

# Create GitHub repository
Write-Host "`n🌐 Creating GitHub repository..." -ForegroundColor Yellow
$repoName = "omni-dev-agent"
$description = "AI-powered development planning and project management tool with intelligent task breakdown and effort estimation"

try {
    gh repo create $repoName --public --description $description --source . --push
    Write-Host "✅ Repository created successfully!" -ForegroundColor Green
    
    # Get repository URL
    $repoUrl = gh repo view --json url --jq .url
    Write-Host "`n🎉 Your repository is now live at:" -ForegroundColor Green
    Write-Host $repoUrl -ForegroundColor Cyan
    
    # Create README badge
    Write-Host "`n📋 Adding project badges..." -ForegroundColor Yellow
    
    # Get current username
    $username = gh api user --jq .login
    
    Write-Host "`n✨ Project Successfully Published!" -ForegroundColor Green
    Write-Host "=================================" -ForegroundColor Green
    Write-Host "Repository URL: $repoUrl" -ForegroundColor Cyan
    Write-Host "Clone URL: https://github.com/$username/$repoName.git" -ForegroundColor Cyan
    
    Write-Host "`n🚀 Share your project:" -ForegroundColor Yellow
    Write-Host "LinkedIn: Check out my AI development planning tool!" -ForegroundColor White
    Write-Host "Twitter: Built an intelligent project planning assistant with Python! #AI #Development" -ForegroundColor White
    Write-Host "Email: I've created an automated development planning tool - check it out at $repoUrl" -ForegroundColor White
    
} catch {
    Write-Host "❌ Error creating repository: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "💡 You may need to delete existing repository with same name or choose different name" -ForegroundColor Yellow
}

Write-Host "`n🎯 Next Steps:" -ForegroundColor Green
Write-Host "1. Share the repository URL with others" -ForegroundColor Yellow
Write-Host "2. Add collaborators if needed: gh repo edit --add-collaborator USERNAME" -ForegroundColor Yellow
Write-Host "3. Create issues for future enhancements: gh issue create" -ForegroundColor Yellow
Write-Host "4. Set up GitHub Pages for documentation: gh repo edit --enable-pages" -ForegroundColor Yellow
