#!/usr/bin/env python3
"""
Demo script to show the future forecasting functionality
This creates sample forecast data to demonstrate the new feature
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json

# Sample data to demonstrate future forecasting functionality
def generate_demo_forecast():
    """Generate demo forecast data to show the new functionality"""
    
    # Sample current IRFC.NS data
    current_price = 45.30  # Current IRFC.NS price
    symbol = "IRFC.NS"
    
    # Sample LSTM model forecast (this would come from the trained model)
    lstm_forecast = {
        'current_price': current_price,
        'predicted_price': 47.85,
        'direction': "📈 UP",
        'movement_percent': 5.64,
        'confidence': 73.2,
        'prediction_binary': 1,
        'movement_str': "+5.64%",
        'model_name': "LSTM",
        'symbol': symbol
    }
    
    return lstm_forecast

def create_forecast_visualization():
    """Create a visualization showing the future forecast"""
    
    forecast = generate_demo_forecast()
    
    # Create the forecast card visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Hide axes
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Background color based on prediction
    bg_color = '#d4f7dc' if forecast['prediction_binary'] == 1 else '#f8d7da'
    border_color = '#28a745' if forecast['prediction_binary'] == 1 else '#dc3545'
    
    # Create the forecast card
    from matplotlib.patches import Rectangle
    rect = Rectangle((1, 2), 8, 6, linewidth=3, edgecolor=border_color, 
                    facecolor=bg_color, alpha=0.8)
    ax.add_patch(rect)
    
    # Add title
    ax.text(5, 9, f'{forecast["model_name"]} Future Forecast for {forecast["symbol"]}', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Add prediction direction
    ax.text(5, 7.5, forecast['direction'], fontsize=24, ha='center', fontweight='bold')
    
    # Add current price
    ax.text(5, 6.5, f'Current Price: ₹{forecast["current_price"]:.2f}', 
            fontsize=14, ha='center')
    
    # Add predicted price
    ax.text(5, 5.8, f'Predicted Price: ₹{forecast["predicted_price"]:.2f}', 
            fontsize=14, ha='center', fontweight='bold')
    
    # Add movement
    ax.text(5, 5.1, f'Expected Movement: {forecast["movement_str"]}', 
            fontsize=12, ha='center')
    
    # Add confidence
    ax.text(5, 4.4, f'Model Confidence: {forecast["confidence"]:.1f}%', 
            fontsize=12, ha='center')
    
    # Add timestamp
    ax.text(5, 3.5, f'Forecast Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
            fontsize=10, ha='center', style='italic')
    
    # Add next trading period info
    ax.text(5, 2.8, 'Prediction for next trading period', 
            fontsize=10, ha='center', style='italic', color='gray')
    
    plt.title('🔮 IRFC.NS Future Price Forecasting - LSTM Model', fontsize=18, pad=20)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('/home/runner/work/xdj/xdj/future_forecast_demo.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✅ Future forecast visualization saved as 'future_forecast_demo.png'")
    return forecast

def create_consensus_forecast():
    """Create a demonstration of consensus forecasting with multiple models"""
    
    # Sample forecasts from multiple models
    forecasts = [
        {
            'model_name': 'LSTM',
            'predicted_price': 47.85,
            'confidence': 73.2,
            'prediction_binary': 1,
            'direction': '📈 UP'
        },
        {
            'model_name': 'PLSTM-TAL', 
            'predicted_price': 46.90,
            'confidence': 68.7,
            'prediction_binary': 1,
            'direction': '📈 UP'
        },
        {
            'model_name': 'CNN',
            'predicted_price': 44.80,
            'confidence': 61.3,
            'prediction_binary': 0,
            'direction': '📉 DOWN'
        }
    ]
    
    current_price = 45.30
    
    # Calculate consensus
    avg_predicted_price = np.mean([f['predicted_price'] for f in forecasts])
    avg_confidence = np.mean([f['confidence'] for f in forecasts])
    up_votes = sum([1 for f in forecasts if f['prediction_binary'] == 1])
    down_votes = len(forecasts) - up_votes
    
    consensus_direction = "📈 UP" if up_votes > down_votes else "📉 DOWN"
    movement_pct = ((avg_predicted_price - current_price) / current_price) * 100
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Individual model forecasts
    models = [f['model_name'] for f in forecasts]
    predictions = [f['predicted_price'] for f in forecasts]
    confidences = [f['confidence'] for f in forecasts]
    colors = ['green' if f['prediction_binary'] == 1 else 'red' for f in forecasts]
    
    ax1.bar(models, predictions, color=colors, alpha=0.7)
    ax1.axhline(y=current_price, color='blue', linestyle='--', label=f'Current Price (₹{current_price})')
    ax1.set_title('Individual Model Forecasts', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Predicted Price (₹)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (pred, conf) in enumerate(zip(predictions, confidences)):
        ax1.text(i, pred + 0.5, f'₹{pred:.2f}\n({conf:.1f}%)', 
                ha='center', va='bottom', fontsize=10)
    
    # Consensus forecast
    ax2.pie([up_votes, down_votes], labels=[f'UP ({up_votes})', f'DOWN ({down_votes})'], 
           colors=['lightgreen', 'lightcoral'], autopct='%1.0f%%', startangle=90)
    ax2.set_title('Model Consensus', fontsize=14, fontweight='bold')
    
    # Add consensus info
    fig.suptitle(f'🔮 IRFC.NS Multi-Model Consensus Forecast\n'
                f'Consensus: {consensus_direction} | Avg Price: ₹{avg_predicted_price:.2f} | '
                f'Movement: {movement_pct:+.2f}% | Avg Confidence: {avg_confidence:.1f}%', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/runner/work/xdj/xdj/consensus_forecast_demo.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✅ Consensus forecast visualization saved as 'consensus_forecast_demo.png'")
    
    return {
        'individual_forecasts': forecasts,
        'consensus': {
            'direction': consensus_direction,
            'avg_predicted_price': avg_predicted_price,
            'movement_percent': movement_pct,
            'avg_confidence': avg_confidence,
            'up_votes': up_votes,
            'down_votes': down_votes
        }
    }

def main():
    """Main demo function"""
    print("🔮 Generating Future Forecasting Demo...")
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)
    
    # Generate individual forecast
    forecast = create_forecast_visualization()
    print(f"📈 {forecast['model_name']} Forecast for {forecast['symbol']}:")
    print(f"   Current Price: ₹{forecast['current_price']:.2f}")
    print(f"   Predicted Price: ₹{forecast['predicted_price']:.2f}")
    print(f"   Direction: {forecast['direction']}")
    print(f"   Movement: {forecast['movement_str']}")
    print(f"   Confidence: {forecast['confidence']:.1f}%")
    print()
    
    # Generate consensus forecast
    consensus_data = create_consensus_forecast()
    consensus = consensus_data['consensus']
    print(f"🎯 Multi-Model Consensus:")
    print(f"   Consensus Direction: {consensus['direction']}")
    print(f"   Average Predicted Price: ₹{consensus['avg_predicted_price']:.2f}")
    print(f"   Expected Movement: {consensus['movement_percent']:+.2f}%")
    print(f"   Average Confidence: {consensus['avg_confidence']:.1f}%")
    print(f"   Model Voting: {consensus['up_votes']} UP, {consensus['down_votes']} DOWN")
    print()
    
    print("✅ Demo completed! Check the generated PNG files for visualizations.")
    
    # Save the forecast data
    demo_data = {
        'timestamp': datetime.now().isoformat(),
        'symbol': 'IRFC.NS',
        'individual_forecast': forecast,
        'consensus_forecast': consensus_data
    }
    
    with open('/home/runner/work/xdj/xdj/demo_forecast_data.json', 'w') as f:
        json.dump(demo_data, f, indent=2)
    
    print("💾 Forecast data saved to 'demo_forecast_data.json'")

if __name__ == "__main__":
    main()