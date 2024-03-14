import React, { useState, useEffect } from 'react';
import axios from 'axios';
import API_BASE_URL from "../config";


const ClosetCraftApp = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [outfitRecommendation, setOutfitRecommendation] = useState({ category: '', color: '' });
  const [data, setData] = useState('');

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
  
    if (file) {
      const formData = new FormData();
      formData.append('file', file); 
  
      try {
        const response = await axios.post(`${API_BASE_URL}/get_recommendation`, formData);
        console.log(response)

        if (response.data && response.data.category) {
          setOutfitRecommendation({ category: response.data.category, color: response.data.color });
        }
  
        setSelectedImage(URL.createObjectURL(file));
      } catch (error) {
        console.error('Error uploading image:', error);
      }
    } else {
      console.error('No file selected');
    }
  };
  

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/profile`);
        setData(response.data);
      } catch (error) {
        console.error("Error fetching data:", error);
      }
    };

    fetchData();
  }, []);

  return (
    <div>
      <h1>ClosetCraft - Clothing Identifier and Color Recommender</h1>
      <div>
        <h1>Data from Flask Backend:</h1>
        {data && <p>{data.message}</p>}
    </div>
      <input type="file" accept="image/*" onChange={handleImageUpload} />
      {selectedImage && (
        <div>
          <h2>Uploaded Image:</h2>
          <img src={selectedImage} alt="Uploaded Outfit" style={{ maxWidth: '100%' }} />
        </div>
      )}
      {outfitRecommendation.category && (
        <div>
          <h2>Outfit Recommendation:</h2>
          <p>Category: {outfitRecommendation.category}</p>
          <p>Color: {outfitRecommendation.color}</p>
        </div>
      )}
    </div>
  );
};

export default ClosetCraftApp;
