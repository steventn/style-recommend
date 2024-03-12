import React, { useState, useEffect } from 'react';
import axios from 'axios';
import API_BASE_URL from "../config";


const ClosetCraftApp = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [outfitRecommendation, setOutfitRecommendation] = useState('');
  const [data, setData] = useState('');

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];

    if (file) {
      const formData = new FormData();
      formData.append('image', file);

      try {
        const response = await axios.post('YOUR_UPLOAD_ENDPOINT', formData);

        if (response.data && response.data.recommendation) {
          setOutfitRecommendation(response.data.recommendation);
        }

        setSelectedImage(URL.createObjectURL(file));
      } catch (error) {
        console.error('Error uploading image:', error);
      }
    }
  };

  useEffect(() => {
    // Fetch data from the Flask backend
    fetch(`${API_BASE_URL}/profile`)
      .then(response => response.json())
      .then(result => setData(result))
      .catch(error => console.error("Error fetching data:", error));
  }, []);

  return (
    <div>
      <h1>ClosetCraft - Outfit Recommender</h1>
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
      {outfitRecommendation && (
        <div>
          <h2>Outfit Recommendation:</h2>
          <p>{outfitRecommendation}</p>
        </div>
      )}
    </div>
  );
};

export default ClosetCraftApp;
