"""
Main FastAPI Application
Agriculture Disease Detection API for Indian Farmers
"""
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import settings
from app.routers import disease_detection, health
from app.services.model_loader import model_loader
from app.services.stage1_crop_service import crop_service
from app.services.stage2_disease_service import disease_service
from app.services.treatment_service import treatment_service
from app.utils.logger import setup_logger

# Setup logging
setup_logger()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events
    Load models on startup, cleanup on shutdown
    """
    # Startup
    logger.info("="*60)
    logger.info("STARTING AGRICULTURE DISEASE DETECTION API")
    logger.info("="*60)
    
    try:
        # Load all ML models
        logger.info("Loading machine learning models...")
        model_loader.load_all_models()
        
        # Initialize services
        logger.info("Initializing services...")
        crop_service.initialize()
        disease_service.initialize()
        
        logger.info("✓ All models and services initialized successfully!")
        logger.info(f"✓ API is ready to serve requests on {settings.HOST}:{settings.PORT}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"✗ Failed to initialize application: {str(e)}")
        sys.exit(1)
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")
    logger.info("Cleanup completed")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    ## कृषि रोग पहचान API | Agriculture Disease Detection API
    
    **किसानों के लिए सरल और प्रभावी फसल रोग पहचान प्रणाली**
    
    ### Features | विशेषताएं:
    
    - 🌾 **9 फसलों का समर्थन** | Support for 9 major crops
      - गन्ना (Sugarcane), मक्का (Maize), गेहूं (Wheat)
      - बाजरा (Bajra), रागी (Ragi), कपास (Cotton)
      - जूट (Jute), जौ (Barley), मटर (Pea)
    
    - 🔬 **दो-चरण पहचान प्रणाली** | Two-stage detection system
      - चरण 1: फसल की पहचान (97%+ सटीकता)
      - चरण 2: रोग की पहचान (विशिष्ट मॉडल)
    
    - 💊 **किसान-अनुकूल उपचार** | Farmer-friendly treatments
      - घरेलू नुस्खे (सस्ते और आसान)
      - रासायनिक उपचार (छोटे और बड़े खेतों के लिए)
      - रोकथाम के उपाय
    
    - 🌐 **भाषा समर्थन** | Language Support
      - हिंदी (Hindi) - मुख्य
      - English
    
    ### Usage | उपयोग:
    
    1. पत्ती की साफ तस्वीर लें | Take clear leaf photo
    2. `/detect-disease` endpoint पर अपलोड करें | Upload to endpoint
    3. रोग और उपचार जानकारी प्राप्त करें | Get disease & treatment info
    
    ### Support | सहायता:
    
    - 📞 किसान कॉल सेंटर: **1800-180-1551**
    - 📧 Email: support@agritech.gov.in
    
    ---
    
    **Developed for Indian Farmers | भारतीय किसानों के लिए विकसित**
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom Exception Handlers

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Validation error",
            "details": exc.errors(),
            "message_hindi": "अमान्य डेटा। कृपया सही जानकारी दें।"
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "message": str(exc),
            "message_hindi": "सर्वर में त्रुटि। कृपया बाद में पुनः प्रयास करें।"
        }
    )


# Include Routers
app.include_router(
    health.router,
    tags=["Health Check"]
)

app.include_router(
    disease_detection.router,
    prefix="/api/v1",
    tags=["Disease Detection"]
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests"""
    logger.info(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
