"""
Treatment and Prevention Recommendations Service
Provides farmer-friendly treatment information
"""
import json
import logging
from typing import Dict, Any, Optional
from app.config import settings

logger = logging.getLogger(__name__)


class TreatmentService:
    """Service for disease treatment recommendations"""
    
    def __init__(self):
        self.treatment_database = {}
        self.load_treatment_database()
    
    def load_treatment_database(self):
        """Load treatment database from JSON file"""
        try:
            with open(settings.TREATMENT_DB_PATH, 'r', encoding='utf-8') as f:
                self.treatment_database = json.load(f)
            logger.info("Treatment database loaded successfully")
        except Exception as e:
            logger.error(f"Error loading treatment database: {str(e)}")
            self.treatment_database = {}
    
    def get_treatment_recommendations(
        self,
        crop_type: str,
        disease_name: str,
        severity: str,
        field_size: str = "small"
    ) -> Dict[str, Any]:
        """
        Get treatment recommendations for a disease
        
        Args:
            crop_type: Type of crop
            disease_name: Name of disease
            severity: Disease severity ('low', 'moderate', 'high')
            field_size: Farm size ('small' or 'large')
            
        Returns:
            Treatment recommendations in farmer-friendly format
        """
        try:
            # Check if crop exists in database
            if crop_type not in self.treatment_database:
                return self._get_default_recommendations(crop_type, disease_name)
            
            crop_data = self.treatment_database[crop_type]
            
            # Check if disease exists
            if disease_name not in crop_data:
                return self._get_default_recommendations(crop_type, disease_name)
            
            disease_data = crop_data[disease_name]
            
            # Build comprehensive recommendation
            recommendations = {
                'disease_name_hindi': disease_data.get('disease_name_hindi', disease_name),
                'disease_name_english': disease_data.get('disease_name_english', disease_name),
                'symptoms': disease_data.get('symptoms', []),
                'home_remedies': disease_data.get('home_remedies', []),
                'chemical_treatment': disease_data.get('chemical_treatment', {}).get(
                    f'{field_size}_field',
                    {}
                ),
                'prevention': disease_data.get('prevention', []),
                'severity_action': disease_data.get('severity_actions', {}).get(severity, ''),
                'expert_contact': disease_data.get('expert_contact', 'किसान कॉल सेंटर: 1800-180-1551'),
                'urgent_action_required': severity == 'high'
            }
            
            # Add severity-based recommendations
            recommendations['recommended_approach'] = self._get_approach_by_severity(
                severity,
                disease_data
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting treatment recommendations: {str(e)}")
            return self._get_default_recommendations(crop_type, disease_name)
    
    def _get_approach_by_severity(self, severity: str, disease_data: Dict) -> str:
        """Get recommended treatment approach based on severity"""
        approaches = {
            'low': 'पहले घरेलू उपाय आजमाएं। अगर 7 दिन में सुधार न हो तो रासायनिक उपचार करें।',
            'moderate': 'घरेलू और रासायनिक दोनों उपचार साथ में करें। नियमित निगरानी रखें।',
            'high': 'तुरंत रासायनिक उपचार शुरू करें। कृषि विशेषज्ञ से संपर्क करें। देरी न करें।'
        }
        
        return approaches.get(severity, 'कृषि विशेषज्ञ से परामर्श लें।')
    
    def _get_default_recommendations(self, crop_type: str, disease_name: str) -> Dict[str, Any]:
        """Provide default recommendations when specific treatment not found"""
        return {
            'disease_name_hindi': disease_name,
            'disease_name_english': disease_name,
            'symptoms': ['विशिष्ट लक्षणों की जानकारी उपलब्ध नहीं है'],
            'home_remedies': [
                {
                    'treatment': 'नीम का तेल स्प्रे',
                    'ingredients': '5 मिली नीम का तेल + 1 लीटर पानी',
                    'preparation': 'नीम के तेल को पानी में अच्छे से मिलाएं',
                    'application': 'पत्तियों पर छिड़काव करें',
                    'cost_estimate': '₹30-50 प्रति एकड़'
                }
            ],
            'chemical_treatment': {
                'product': 'कृषि विशेषज्ञ से सलाह लें',
                'dosage': 'विशेषज्ञ की सिफारिश के अनुसार',
                'method': 'विशेषज्ञ मार्गदर्शन में',
                'cost': 'परिवर्तनीय'
            },
            'prevention': [
                'स्वस्थ बीज का उपयोग करें',
                'खेत की साफ-सफाई रखें',
                'उचित जल निकासी सुनिश्चित करें',
                'नियमित निगरानी करें'
            ],
            'severity_action': 'कृषि विशेषज्ञ या किसान कॉल सेंटर से संपर्क करें',
            'expert_contact': 'किसान कॉल सेंटर: 1800-180-1551',
            'recommended_approach': 'सटीक उपचार के लिए नजदीकी कृषि विज्ञान केंद्र (KVK) से संपर्क करें।',
            'urgent_action_required': False
        }
    
    def get_general_prevention_tips(self, crop_type: str) -> Dict[str, Any]:
        """
        Get general prevention tips for a crop
        
        Args:
            crop_type: Type of crop
            
        Returns:
            General prevention guidelines
        """
        general_tips = {
            'सामान्य रोकथाम के उपाय': [
                'रोग प्रतिरोधी किस्मों का चयन करें',
                'बीज उपचार अवश्य करें',
                'खेत में फसल अवशेष न छोड़ें',
                'संतुलित उर्वरक का प्रयोग करें',
                'नियमित निगरानी करते रहें',
                'पौधों के बीच उचित दूरी रखें',
                'जल निकासी का प्रबंधन करें',
                'फसल चक्र अपनाएं'
            ],
            'जैविक खेती के उपाय': [
                'नीम की खली का प्रयोग करें',
                'गोमूत्र का छिड़काव करें',
                'जैविक खाद का उपयोग करें',
                'ट्राइकोडर्मा का प्रयोग करें',
                'वर्मीकम्पोस्ट डालें'
            ],
            'तत्काल कार्रवाई संकेत': [
                'पत्तियों पर असामान्य धब्बे दिखें',
                'पौधे की बढ़वार रुक जाए',
                'पत्तियां मुड़ने या सूखने लगें',
                'फल या फूल समय से पहले गिरने लगें'
            ]
        }
        
        return general_tips


# Global service instance
treatment_service = TreatmentService()
