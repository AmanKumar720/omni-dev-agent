# GitHub Personal Access Token Setup Script
# This script helps you create and configure a GitHub PAT with required scopes

Write-Host "GitHub Personal Access Token Setup" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Write-Host ""

Write-Host "Required Scopes for this project:" -ForegroundColor Yellow
Write-Host "  - repo (Full control of private repositories)" -ForegroundColor White
Write-Host "  - workflow (Update GitHub Action workflows)" -ForegroundColor White
Write-Host "  - admin:org (Full control of orgs and teams, read and write org projects)" -ForegroundColor White
Write-Host ""

Write-Host "Steps to create your GitHub PAT:" -ForegroundColor Cyan
Write-Host "1. Go to https://github.com/settings/tokens/new" -ForegroundColor White
Write-Host "2. Give your token a descriptive name (e.g., 'omni-dev-agent-automation')" -ForegroundColor White
Write-Host "3. Set expiration (recommended: 90 days)" -ForegroundColor White
Write-Host "4. Select the following scopes:" -ForegroundColor White
Write-Host "   [x] repo" -ForegroundColor Green
Write-Host "   [x] workflow" -ForegroundColor Green
Write-Host "   [x] admin:org" -ForegroundColor Green
Write-Host "5. Click 'Generate token'" -ForegroundColor White
Write-Host "6. Copy the token (you won't see it again!)" -ForegroundColor White
Write-Host ""

Write-Host "After creating your token, run:" -ForegroundColor Magenta
Write-Host "  `$env:GITHUB_TOKEN = 'your_token_here'" -ForegroundColor Yellow
Write-Host ""
Write-Host "Or for persistent storage:" -ForegroundColor Magenta
Write-Host "  [Environment]::SetEnvironmentVariable('GITHUB_TOKEN', 'your_token_here', 'User')" -ForegroundColor Yellow
Write-Host ""

# Check if token is already set
if ($env:GITHUB_TOKEN) {
    Write-Host "[OK] GITHUB_TOKEN is already set in your environment" -ForegroundColor Green
    
    # Test the token
    Write-Host "Testing token validity..." -ForegroundColor Yellow
    try {
        $headers = @{
            'Authorization' = "token $env:GITHUB_TOKEN"
            'Accept' = 'application/vnd.github.v3+json'
            'User-Agent' = 'omni-dev-agent-setup'
        }
        
        $response = Invoke-RestMethod -Uri "https://api.github.com/user" -Headers $headers
        Write-Host "[OK] Token is valid! Authenticated as: $($response.login)" -ForegroundColor Green
        
        # Check scopes
        $scopeResponse = Invoke-WebRequest -Uri "https://api.github.com/user" -Headers $headers
        $scopes = $scopeResponse.Headers['X-OAuth-Scopes'] -split ', '
        
        Write-Host "Current token scopes: $($scopes -join ', ')" -ForegroundColor Cyan
        
        $requiredScopes = @('repo', 'workflow', 'admin:org')
        $missingScopes = @()
        
        foreach ($scope in $requiredScopes) {
            if ($scope -notin $scopes) {
                $missingScopes += $scope
            }
        }
        
        if ($missingScopes.Count -eq 0) {
            Write-Host "[OK] All required scopes are present!" -ForegroundColor Green
        } else {
            Write-Host "[WARN] Missing required scopes: $($missingScopes -join ', ')" -ForegroundColor Red
            Write-Host "Please create a new token with the missing scopes." -ForegroundColor Yellow
        }
        
    } catch {
        Write-Host "[ERROR] Token test failed: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Please check your token and try again." -ForegroundColor Yellow
    }
} else {
    Write-Host "[WARN] GITHUB_TOKEN not found in environment variables" -ForegroundColor Yellow
    Write-Host "Please create and set your GitHub Personal Access Token using the instructions above." -ForegroundColor White
}

Write-Host ""
Write-Host "Configuration Summary:" -ForegroundColor Green
Write-Host "Repository: AmanKumar720/omni-dev-agent" -ForegroundColor White
Write-Host "Default Branch: main" -ForegroundColor White
Write-Host "GitHub API: https://api.github.com" -ForegroundColor White
