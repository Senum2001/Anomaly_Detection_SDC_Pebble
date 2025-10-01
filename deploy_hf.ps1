# Hugging Face Spaces Deployment Script
# This script helps you deploy to Hugging Face Spaces

Write-Host "🚀 Hugging Face Spaces Deployment Helper" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Git is installed
try {
    git --version | Out-Null
} catch {
    Write-Host "❌ Git is not installed. Please install Git first." -ForegroundColor Red
    exit 1
}

Write-Host "📋 Pre-deployment Checklist:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Create Hugging Face account at https://huggingface.co/join" -ForegroundColor White
Write-Host "2. Create new Space at https://huggingface.co/new-space" -ForegroundColor White
Write-Host "   - Name: anomaly-detection-api (or your choice)" -ForegroundColor Gray
Write-Host "   - SDK: Docker" -ForegroundColor Gray
Write-Host "   - Hardware: CPU basic (free)" -ForegroundColor Gray
Write-Host "3. Get your Space URL from the dashboard" -ForegroundColor White
Write-Host ""

# Ask for HF Space details
$username = Read-Host "Enter your Hugging Face username"
$spaceName = Read-Host "Enter your Space name (e.g., anomaly-detection-api)"

$hfSpaceUrl = "https://huggingface.co/spaces/$username/$spaceName"

Write-Host ""
Write-Host "📦 Preparing deployment to: $hfSpaceUrl" -ForegroundColor Green
Write-Host ""

# Create a deployment directory
$deployDir = "hf_deploy"
if (Test-Path $deployDir) {
    Write-Host "⚠️  Deployment directory already exists. Cleaning..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $deployDir
}

Write-Host "📁 Creating deployment directory..." -ForegroundColor Cyan
New-Item -ItemType Directory -Path $deployDir | Out-Null

# Copy necessary files
Write-Host "📋 Copying files..." -ForegroundColor Cyan

$filesToCopy = @(
    "app.py",
    "inference_core.py",
    "requirements.txt",
    "README.md"
)

$directoriesToCopy = @(
    "scripts",
    "configs"
)

foreach ($file in $filesToCopy) {
    if (Test-Path $file) {
        Copy-Item $file $deployDir\ -Force
        Write-Host "  ✓ $file" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $file (not found)" -ForegroundColor Red
    }
}

foreach ($dir in $directoriesToCopy) {
    if (Test-Path $dir) {
        Copy-Item -Recurse $dir $deployDir\ -Force
        Write-Host "  ✓ $dir\" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $dir\ (not found)" -ForegroundColor Yellow
    }
}

# Copy and rename Dockerfile.hf to Dockerfile
if (Test-Path "Dockerfile.hf") {
    Copy-Item "Dockerfile.hf" "$deployDir\Dockerfile" -Force
    Write-Host "  ✓ Dockerfile.hf → Dockerfile" -ForegroundColor Green
} else {
    Write-Host "  ✗ Dockerfile.hf (not found)" -ForegroundColor Red
}

Write-Host ""
Write-Host "✅ Files prepared in $deployDir\" -ForegroundColor Green
Write-Host ""
Write-Host "📝 Next Steps:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Clone your Hugging Face Space:" -ForegroundColor White
Write-Host "   cd .." -ForegroundColor Gray
Write-Host "   git clone https://huggingface.co/spaces/$username/$spaceName" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Copy files to your Space:" -ForegroundColor White
Write-Host "   Copy-Item -Recurse AI\$deployDir\* $spaceName\" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Push to Hugging Face:" -ForegroundColor White
Write-Host "   cd $spaceName" -ForegroundColor Gray
Write-Host "   git add ." -ForegroundColor Gray
Write-Host "   git commit -m 'Initial deployment'" -ForegroundColor Gray
Write-Host "   git push" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Wait for build (5-10 minutes)" -ForegroundColor White
Write-Host ""
Write-Host "5. Access your API at:" -ForegroundColor White
Write-Host "   https://$username-$(($spaceName -replace '_','-')).hf.space" -ForegroundColor Cyan
Write-Host ""

# Ask if user wants to continue with automated setup
$continue = Read-Host "Do you want to continue with automated Git setup? (y/n)"

if ($continue -eq "y" -or $continue -eq "Y") {
    Write-Host ""
    Write-Host "🔧 Setting up Git repository..." -ForegroundColor Cyan
    
    $hfToken = Read-Host "Enter your Hugging Face token (from https://huggingface.co/settings/tokens)" -AsSecureString
    $hfTokenPlain = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($hfToken)
    )
    
    Set-Location ..
    
    # Clone the space
    Write-Host "Cloning your Space..." -ForegroundColor Cyan
    git clone "https://$username:$hfTokenPlain@huggingface.co/spaces/$username/$spaceName"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Space cloned successfully" -ForegroundColor Green
        
        # Copy files
        Write-Host "Copying files to Space..." -ForegroundColor Cyan
        Copy-Item -Recurse "AI\$deployDir\*" "$spaceName\" -Force
        
        # Git operations
        Set-Location $spaceName
        Write-Host "Committing changes..." -ForegroundColor Cyan
        git add .
        git commit -m "Deploy Anomaly Detection API to HF Spaces"
        
        Write-Host "Pushing to Hugging Face..." -ForegroundColor Cyan
        git push
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "🎉 Deployment successful!" -ForegroundColor Green
            Write-Host ""
            Write-Host "Your API will be available at:" -ForegroundColor Cyan
            Write-Host "https://$username-$(($spaceName -replace '_','-')).hf.space" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "Check build status at:" -ForegroundColor Yellow
            Write-Host "$hfSpaceUrl" -ForegroundColor Yellow
        } else {
            Write-Host "❌ Push failed. Please check your token and try again." -ForegroundColor Red
        }
        
        Set-Location ..\AI
    } else {
        Write-Host "❌ Clone failed. Please check your credentials." -ForegroundColor Red
        Set-Location AI
    }
} else {
    Write-Host ""
    Write-Host "✅ Files ready in $deployDir\" -ForegroundColor Green
    Write-Host "Follow the manual steps above to complete deployment." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Done! 🚀" -ForegroundColor Cyan
