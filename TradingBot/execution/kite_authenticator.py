"""
Kite authentication with daily TOTP prompt for enhanced security.

This module handles OAuth authentication with Zerodha Kite API and requires
TOTP (Time-based One-Time Password) to be entered daily for security.
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import pyotp
from loguru import logger
from kiteconnect import KiteConnect


class KiteAuthenticator:
    """
    Handles Kite API authentication with daily TOTP verification.
    
    Features:
    - OAuth flow with request token
    - TOTP-based 2FA (prompted daily)
    - Token caching with expiry
    - Automatic re-authentication
    """
    
    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize authenticator.
        
        Args:
            api_key: Kite API key
            api_secret: Kite API secret
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.kite = KiteConnect(api_key=api_key)
        self.token_file = Path("config/.kite_token.json")
        
        logger.info("Initialized Kite authenticator")
    
    def _is_token_valid(self) -> bool:
        """
        Check if cached access token is still valid.
        
        Returns:
            True if token exists and hasn't expired
        """
        if not self.token_file.exists():
            return False
        
        try:
            with open(self.token_file, 'r') as f:
                token_data = json.load(f)
            
            # Check if token was created today
            token_date = datetime.fromisoformat(token_data['created_at'])
            today = datetime.now().date()
            
            # Token expires daily at market close (3:30 PM)
            market_close = datetime.combine(today, datetime.min.time().replace(hour=15, minute=30))
            
            if token_date.date() == today and datetime.now() < market_close:
                logger.info("Cached access token is still valid")
                return True
            
            logger.info("Access token has expired (daily expiry)")
            return False
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Invalid token file: {e}")
            return False
    
    def _save_token(self, access_token: str):
        """
        Save access token to cache file.
        
        Args:
            access_token: Kite access token
        """
        token_data = {
            'access_token': access_token,
            'created_at': datetime.now().isoformat(),
            'expires_at': datetime.now().replace(hour=15, minute=30, second=0).isoformat()
        }
        
        # Create config directory if it doesn't exist
        self.token_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.token_file, 'w') as f:
            json.dump(token_data, f, indent=2)
        
        logger.info(f"Saved access token to {self.token_file}")
    
    def _load_token(self) -> str:
        """
        Load access token from cache file.
        
        Returns:
            Access token string
        """
        with open(self.token_file, 'r') as f:
            token_data = json.load(f)
        
        return token_data['access_token']
    
    def _get_totp_code(self) -> str:
        """
        Get TOTP code from user (either generate from secret or enter manually).
        
        Returns:
            6-digit TOTP code
        """
        print("\n" + "="*80)
        print("KITE API AUTHENTICATION - Daily TOTP Verification")
        print("="*80)
        print("\nChoose authentication method:")
        print("1. Enter TOTP secret (will auto-generate code)")
        print("2. Enter TOTP code manually from your authenticator app")
        
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "1":
            print("\nEnter your TOTP secret (long alphanumeric string like 'JBSWY3DPEHPK3PXP'):")
            totp_secret = input("TOTP Secret: ").strip()
            
            if not totp_secret:
                raise ValueError("TOTP secret is required")
            
            try:
                totp = pyotp.TOTP(totp_secret)
                code = totp.now()
                print(f"\n✅ Generated TOTP code: {code}")
                return code
            except Exception as e:
                raise ValueError(f"Invalid TOTP secret: {e}")
        
        elif choice == "2":
            print("\nOpen your authenticator app (Google Authenticator, Authy, etc.)")
            print("and enter the 6-digit code for Zerodha Kite:")
            code = input("TOTP Code: ").strip()
            
            if not code or len(code) != 6 or not code.isdigit():
                raise ValueError("TOTP code must be exactly 6 digits")
            
            print(f"\n✅ Using TOTP code: {code}")
            return code
        
        else:
            raise ValueError("Invalid choice. Enter 1 or 2")
    
    def authenticate(self, request_token: str = None) -> str:
        """
        Authenticate with Kite API.
        
        Flow:
        1. Check if cached token is valid
        2. If not, prompt for TOTP secret (daily)
        3. Generate login URL
        4. User authorizes and provides request token
        5. Exchange request token for access token
        6. Cache access token
        
        Args:
            request_token: Optional request token from OAuth callback
            
        Returns:
            Access token string
        """
        # Check cached token first
        if self._is_token_valid():
            access_token = self._load_token()
            self.kite.set_access_token(access_token)
            logger.info("Using cached access token")
            return access_token
        
        # Need fresh authentication
        logger.info("Cached token invalid or expired, initiating fresh authentication")
        
        # Get TOTP code (either from secret or manually)
        totp_code = self._get_totp_code()
        
        # Generate login URL
        login_url = self.kite.login_url()
        
        print("="*80)
        print("STEP 1: Authorize the app")
        print("="*80)
        print(f"\n1. Open this URL in your browser:\n\n{login_url}\n")
        print(f"2. Login with your Kite credentials")
        print(f"3. Enter TOTP code when prompted: {totp_code}")
        print(f"4. After authorization, you'll be redirected to a URL like:")
        print(f"   http://127.0.0.1/?request_token=XXXXXX&action=login&status=success")
        print(f"5. Copy the 'request_token' value from the URL\n")
        
        # Get request token from user
        if not request_token:
            request_token = input("Enter the request_token from the redirect URL: ").strip()
        
        if not request_token:
            raise ValueError("Request token is required")
        
        # Exchange request token for access token
        try:
            data = self.kite.generate_session(request_token, api_secret=self.api_secret)
            access_token = data['access_token']
            
            # Save token
            self._save_token(access_token)
            self.kite.set_access_token(access_token)
            
            print("\n" + "="*80)
            print("✅ AUTHENTICATION SUCCESSFUL")
            print("="*80)
            print(f"Access token valid until: {datetime.now().replace(hour=15, minute=30).strftime('%Y-%m-%d %I:%M %p')}")
            print("You won't need to re-authenticate until tomorrow.\n")
            
            logger.info("Successfully authenticated with Kite API")
            return access_token
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise
    
    def get_kite_client(self) -> KiteConnect:
        """
        Get authenticated Kite client.
        
        Returns:
            KiteConnect instance with valid access token
        """
        if not self._is_token_valid():
            self.authenticate()
        
        return self.kite


def main():
    """
    Main authentication flow for standalone usage.
    """
    # Load API credentials from environment
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('KITE_API_KEY')
    api_secret = os.getenv('KITE_API_SECRET')
    
    if not api_key or not api_secret:
        print("❌ Error: KITE_API_KEY and KITE_API_SECRET must be set in .env file")
        return
    
    # Authenticate
    authenticator = KiteAuthenticator(api_key, api_secret)
    
    try:
        access_token = authenticator.authenticate()
        
        # Test API call
        kite = authenticator.get_kite_client()
        profile = kite.profile()
        
        print("\n" + "="*80)
        print("PROFILE INFORMATION")
        print("="*80)
        print(f"User ID: {profile['user_id']}")
        print(f"Name: {profile['user_name']}")
        print(f"Email: {profile['email']}")
        print(f"Broker: {profile['broker']}")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Authentication failed: {e}")
        logger.error(f"Authentication error: {e}")


if __name__ == "__main__":
    main()
