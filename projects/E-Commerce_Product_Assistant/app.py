# app.py - CLEAN VERSION WITHOUT CATEGORY DROPDOWN
# ========== ADD THESE AT THE VERY TOP ==========
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
# ===============================================

import streamlit as st
import sys
sys.path.append('src')

import json
import pandas as pd
from PIL import Image
import time

# Page configuration
st.set_page_config(
    page_title="Product Search Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Cleaner design
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2563EB;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .product-card {
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: white;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }
    .product-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
        border-color: #2563EB;
    }
    .product-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #2563EB, #7C3AED);
    }
    .price-tag {
        font-size: 1.5rem;
        font-weight: 800;
        color: #059669;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .rating-badge {
        background: linear-gradient(135deg, #F59E0B, #D97706);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        box-shadow: 0 2px 4px rgba(245, 158, 11, 0.3);
    }
    .category-chip {
        background: linear-gradient(135deg, #DBEAFE, #E0F2FE);
        color: #1E40AF;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .brand-tag {
        background: #F3F4F6;
        color: #4B5563;
        padding: 0.3rem 0.8rem;
        border-radius: 12px;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }
    .stock-badge {
        position: absolute;
        top: 1rem;
        right: 1rem;
        background: #10B981;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .action-button {
        background: linear-gradient(135deg, #2563EB, #4F46E5);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        margin: 0.3rem 0;
        text-align: center;
    }
    .action-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(37, 99, 235, 0.3);
    }
    .search-box {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 2rem;
    }
    .filter-section {
        background: #F8FAFC;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .result-count {
        background: linear-gradient(135deg, #6366F1, #8B5CF6);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        display: inline-block;
        font-weight: 600;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Sample products data
SAMPLE_PRODUCTS = [
    {
        "id": "1",
        "name": "Sony WH-1000XM4 Wireless Headphones",
        "description": "Industry-leading noise cancellation with Dual Noise Sensor technology. Perfect for travel and office use.",
        "price": 349.99,
        "category": "Electronics",
        "brand": "Sony",
        "rating": 4.8,
        "stock": 50,
        "image_url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=400",
        "specs": {"color": "black", "battery": "30 hours", "bluetooth": "5.0"}
    },
    {
        "id": "2",
        "name": "Nike Air Zoom Pegasus 38",
        "description": "Responsive running shoes with Zoom Air cushioning for maximum comfort during workouts.",
        "price": 119.99,
        "category": "Sports",
        "brand": "Nike",
        "rating": 4.6,
        "stock": 100,
        "image_url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w-400",
        "specs": {"size": "7-13", "color": "blue/white", "material": "mesh"}
    },
    {
        "id": "3",
        "name": "Hydro Flask Wide Mouth Water Bottle",
        "description": "Insulated stainless steel water bottle keeps drinks cold for 24 hours, hot for 12 hours.",
        "price": 44.95,
        "category": "Outdoor",
        "brand": "Hydro Flask",
        "rating": 4.7,
        "stock": 200,
        "image_url": "https://images.unsplash.com/photo-1523362628745-0c100150b504?w=400",
        "specs": {"capacity": "32oz", "material": "stainless steel"}
    },
    {
        "id": "4",
        "name": "Apple MacBook Pro 14-inch",
        "description": "Professional laptop with M1 Pro chip and Liquid Retina XDR display for creators.",
        "price": 1999.00,
        "category": "Electronics",
        "brand": "Apple",
        "rating": 4.9,
        "stock": 25,
        "image_url": "https://images.unsplash.com/photo-1517336714731-489689fd1ca8?w=400",
        "specs": {"processor": "M1 Pro", "ram": "16GB", "storage": "512GB SSD"}
    },
    {
        "id": "5",
        "name": "Logitech MX Master 3 Mouse",
        "description": "Advanced wireless mouse with Darkfield tracking for use on any surface.",
        "price": 99.99,
        "category": "Electronics",
        "brand": "Logitech",
        "rating": 4.7,
        "stock": 150,
        "image_url": "https://images.unsplash.com/photo-1527864550417-7fd91fc51a46?w=400",
        "specs": {"connectivity": "wireless", "battery": "70 days"}
    },
    {
        "id": "6",
        "name": "Instant Pot Duo 7-in-1",
        "description": "Electric pressure cooker that replaces 7 kitchen appliances. Cooks meals faster.",
        "price": 99.95,
        "category": "Home",
        "brand": "Instant Pot",
        "rating": 4.7,
        "stock": 75,
        "image_url": "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=400",
        "specs": {"capacity": "6 quarts", "functions": "7-in-1"}
    },
    {
        "id": "7",
        "name": "Kindle Paperwhite",
        "description": "Waterproof e-reader with 6.8\" display and adjustable warm light for reading.",
        "price": 139.99,
        "category": "Electronics",
        "brand": "Amazon",
        "rating": 4.5,
        "stock": 120,
        "image_url": "https://images.unsplash.com/photo-1544716278-ca5e3f4abd8c?w=400",
        "specs": {"storage": "8GB", "battery": "weeks", "waterproof": "IPX8"}
    },
    {
        "id": "8",
        "name": "Dyson V11 Cordless Vacuum",
        "description": "Powerful cordless vacuum with intelligent suction and LCD screen.",
        "price": 599.99,
        "category": "Home",
        "brand": "Dyson",
        "rating": 4.8,
        "stock": 40,
        "image_url": "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=400",
        "specs": {"runtime": "60 mins", "dustbin": "0.76L", "weight": "6.68 lbs"}
    }
]

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""
if 'price_range' not in st.session_state:
    st.session_state.price_range = (0, 2500)

# Simple search function
def search_products(query, price_range=(0, 2500), min_rating=0.0, n_results=8):
    """Simple search function without category filter"""
    results = []
    query_lower = query.lower() if query else ""
    
    for product in SAMPLE_PRODUCTS:
        # Apply filters (NO CATEGORY FILTER)
        if not (price_range[0] <= product['price'] <= price_range[1]):
            continue
        if product['rating'] < min_rating:
            continue
        
        # If no query, include all filtered products
        if not query:
            results.append(product)
            continue
        
        # Check for matches
        score = 0
        name_lower = product["name"].lower()
        desc_lower = product["description"].lower()
        category_lower = product["category"].lower()
        brand_lower = product["brand"].lower()
        
        # Exact phrase match
        if query_lower in name_lower:
            score += 3
        elif query_lower in desc_lower:
            score += 2
        
        # Word matches
        query_words = [w for w in query_lower.split() if len(w) > 2]
        for word in query_words:
            if word in name_lower:
                score += 2
            if word in desc_lower:
                score += 1
            if word in category_lower:
                score += 0.5
            if word in brand_lower:
                score += 0.5
        
        if score > 0:
            results.append({**product, "relevance": score})
    
    # Sort by relevance if there's a query
    if query:
        results.sort(key=lambda x: x.get("relevance", 0), reverse=True)
    
    return {
        "query": query,
        "results": results[:n_results],
        "total_results": len(results)
    }

def create_product_card(product):
    """Create HTML for product card"""
    return f"""
    <div class="product-card">
        <div class="stock-badge">{product['stock']} in stock</div>
        <img src="{product['image_url']}" width="100%" style="border-radius: 8px; margin-bottom: 1rem;">
        <h3 style="margin: 0.5rem 0; font-size: 1.1rem;">{product['name']}</h3>
        <div class="price-tag">${product['price']}</div>
        <div class="rating-badge">⭐ {product['rating']}/5</div>
        <div style="margin: 0.8rem 0;">
            <span class="category-chip">📦 {product['category']}</span>
            <span class="brand-tag">🏷️ {product['brand']}</span>
        </div>
        <p style="color: #666; font-size: 0.9rem; margin: 0.5rem 0; line-height: 1.4;">
            {product['description'][:90]}...
        </p>
    </div>
    """

def main():
    # Header with gradient
    st.markdown('<h1 class="main-header">🔍 Product Search Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; margin-bottom: 2rem; font-size: 1.1rem;">Find products using text descriptions or filters</p>', unsafe_allow_html=True)
    
    # Main search box
    with st.container():
        st.markdown('<div class="search-box">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input(
                "**Search Products**",
                placeholder="Enter product name, description, or keywords...",
                value=st.session_state.get('search_query', ''),
                label_visibility="collapsed"
            )
        
        with col2:
            search_btn = st.button(
                "🔍 Search",
                type="primary",
                use_container_width=True,
                key="search_button"
            )
        
        if search_btn and query:
            st.session_state.search_query = query
            with st.spinner("Searching products..."):
                time.sleep(0.5)  # Simulate search delay
                results = search_products(
                    query=query,
                    price_range=st.session_state.price_range
                )
                st.session_state.search_results = results
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Filters in sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Filters")
        
        # Only price range and rating filters (NO CATEGORY FILTER)
        
        # Price range slider with better styling
        st.markdown("**Price Range**")
        price_range = st.slider(
            "",
            0, 2500, 
            value=st.session_state.price_range,
            step=50,
            label_visibility="collapsed"
        )
        # st.markdown(f"<p style='text-align: center; color: #666;'>${price_range[0]} - ${price_range[1]}</p>", unsafe_allow_html=True)
        
        # Rating filter
        min_rating = st.slider(
            "**Minimum Rating**",
            0.0, 5.0, 0.0,
            step=0.5
        )
        
        # Number of results
        n_results = st.select_slider(
            "**Results per page**",
            options=[4, 6, 8, 12],
            value=8
        )
        
        # Apply filters button
        if st.button("Apply Filters", type="secondary", use_container_width=True):
            st.session_state.price_range = price_range
            
            if st.session_state.search_query:
                results = search_products(
                    query=st.session_state.search_query,
                    price_range=price_range,
                    min_rating=min_rating,
                    n_results=n_results
                )
                st.session_state.search_results = results
        
        # Clear all button
        if st.button("Clear All", type="secondary", use_container_width=True):
            st.session_state.search_results = None
            st.session_state.search_query = ""
            st.session_state.price_range = (0, 2500)
            st.rerun()
        
        # Divider
        st.markdown("---")
        
        # Quick actions (simplified)
        st.markdown("### 🚀 Quick Actions")
        
        if st.button("View All Products", use_container_width=True):
            st.session_state.search_query = ""
            st.session_state.search_results = search_products("", n_results=n_results)
        
        if st.button("Top Rated Products", use_container_width=True):
            st.session_state.search_query = ""
            results = search_products("", n_results=n_results)
            # Sort by rating
            results["results"].sort(key=lambda x: x["rating"], reverse=True)
            st.session_state.search_results = results
    
    # Display results or welcome message
    st.markdown("---")
    
    if st.session_state.search_results and st.session_state.search_results['results']:
        results = st.session_state.search_results['results']
        total_found = st.session_state.search_results['total_results']
        
        # Results header
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
            <div>
                <span class="result-count">{len(results)} of {total_found} products</span>
                <h3 style="margin: 0; display: inline-block; margin-left: 1rem;">
                    Results for: <span style="color: #2563EB;">"{st.session_state.search_query}"</span>
                </h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display products in grid
        cols_per_row = 4  # More compact layout
        for i in range(0, len(results), cols_per_row):
            row_cols = st.columns(cols_per_row)
            
            for j in range(cols_per_row):
                if i + j < len(results):
                    product = results[i + j]
                    with row_cols[j]:
                        # Product card
                        card_html = create_product_card(product)
                        st.markdown(card_html, unsafe_allow_html=True)
                        
                        # Action buttons
                        if st.button(
                            "🛒 Add to Cart",
                            key=f"cart_{product['id']}_{i}_{j}",
                            use_container_width=True,
                            type="primary"
                        ):
                            st.success(f"✓ Added **{product['name']}** to cart!")
                            time.sleep(0.5)
                            st.rerun()
                        
                        if st.button(
                            "💖 Wishlist",
                            key=f"wish_{product['id']}_{i}_{j}",
                            use_container_width=True
                        ):
                            st.info(f"✓ Saved **{product['name']}** to wishlist!")
                            time.sleep(0.5)
                            st.rerun()
                        
                        if st.button(
                            "📝 Details",
                            key=f"detail_{product['id']}_{i}_{j}",
                            use_container_width=True
                        ):
                            st.session_state.selected_product = product
                            st.rerun()
    
    elif st.session_state.search_results and not st.session_state.search_results['results']:
        # No results found
        st.warning("No products found matching your criteria.")
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h3 style="color: #666;">Try these suggestions:</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Search suggestions
        sug_cols = st.columns(4)
        suggestions = ["Electronics", "Sports", "Home", "Accessories"]
        for idx, suggestion in enumerate(suggestions):
            with sug_cols[idx]:
                if st.button(suggestion, key=f"sug_{suggestion}", use_container_width=True):
                    st.session_state.search_query = suggestion
                    st.rerun()
    
    else:
        # Welcome screen with featured products
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h2 style="color: #2563EB; margin-bottom: 1rem;">🎯 Find Your Perfect Product</h2>
            <p style="color: #666; font-size: 1.1rem; max-width: 800px; margin: 0 auto 2rem auto;">
                Use the search bar above to find products by name, description, or category.<br>
                Apply price and rating filters to narrow down your search results.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Featured products section
        st.markdown("### 🔥 Featured Products")
        
        # Display featured products
        feat_products = [SAMPLE_PRODUCTS[0], SAMPLE_PRODUCTS[2], SAMPLE_PRODUCTS[4], SAMPLE_PRODUCTS[6]]
        feat_cols = st.columns(4)
        
        for idx, product in enumerate(feat_products):
            with feat_cols[idx]:
                st.image(product['image_url'], use_container_width=True)
                st.markdown(f"**{product['name']}**")
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: bold; color: #059669;">${product['price']}</span>
                    <span style="background: #FBBF24; color: white; padding: 0.2rem 0.6rem; border-radius: 12px;">
                        ⭐ {product['rating']}
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Search Similar", key=f"feat_btn_{product['id']}", use_container_width=True):
                    st.session_state.search_query = product['name']
                    st.rerun()

# Product detail view
def show_product_detail(product):
    """Show detailed product view"""
    st.markdown('<h1 class="main-header">Product Details</h1>', unsafe_allow_html=True)
    
    # Back button
    if st.button("← Back to Search", key="back_button"):
        if 'selected_product' in st.session_state:
            del st.session_state.selected_product
        st.rerun()
    
    # Product details
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(product['image_url'], use_container_width=True)
    
    with col2:
        st.markdown(f"## {product['name']}")
        
        # Price and rating
        col_price, col_rating, col_stock = st.columns(3)
        with col_price:
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="font-size: 0.9rem; color: #666;">Price</div>
                <div class="price-tag" style="justify-content: center;">${product['price']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_rating:
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="font-size: 0.9rem; color: #666;">Rating</div>
                <div class="rating-badge" style="justify-content: center;">⭐ {product['rating']}/5</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_stock:
            stock_color = "#10B981" if product['stock'] > 20 else "#EF4444"
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="font-size: 0.9rem; color: #666;">Stock</div>
                <div style="font-size: 1.3rem; font-weight: bold; color: {stock_color};">
                    {product['stock']} units
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Category and brand
        st.markdown(f"""
        <div style="margin: 1rem 0;">
            <span class="category-chip" style="font-size: 1rem;">📦 {product['category']}</span>
            <span class="brand-tag" style="font-size: 1rem;">🏷️ {product['brand']}</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📋 Description")
        st.write(product['description'])
        
        st.markdown("### 🔧 Specifications")
        specs = product.get('specs', {})
        for key, value in specs.items():
            st.markdown(f"- **{key.replace('_', ' ').title()}:** {value}")
    
    # Action buttons
    st.markdown("---")
    st.markdown("### 🛒 Actions")
    
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        if st.button("🛒 Add to Cart", type="primary", use_container_width=True):
            st.success(f"✓ Added {product['name']} to cart!")
    
    with btn_col2:
        if st.button("💖 Add to Wishlist", use_container_width=True):
            st.info(f"✓ Added {product['name']} to wishlist!")
    
    with btn_col3:
        if st.button("📞 Contact Seller", use_container_width=True):
            st.info("Seller contact information would appear here.")

# Run the app
if __name__ == "__main__":
    if 'selected_product' in st.session_state:
        show_product_detail(st.session_state.selected_product)
    else:
        main()
    
    # Simple footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 1rem;'>"
        "Product Search Assistant • v1.0 • Streamlit App"
        "</div>",
        unsafe_allow_html=True
    )