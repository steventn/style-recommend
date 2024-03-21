import React, { useState } from 'react';
import Header from './components/Header/Header.js';
import ImageUpload from './components/ImageUpload/ImageUpload.js';
import styles from './components/ImageUpload/ImageUpload.module.css';
import ColorRecommendation from './components/ColorRecommendation/ColorRecommendation.js';
import API_BASE_URL from "./config";
import axios from 'axios';

const App = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [outfitRecommendation, setOutfitRecommendation] = useState({ category: '', primary_color: '', color: {} });

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    setSelectedImage(URL.createObjectURL(file));

    if (file) {
      const formData = new FormData();
      formData.append('file', file); 
  
      try {
        const response = await axios.post(`${API_BASE_URL}/get_recommendation`, formData);

        if (response.data && response.data.category) {
          setOutfitRecommendation(response.data);
        }
  
        setSelectedImage(URL.createObjectURL(file));
      } catch (error) {
        console.error('Error uploading image:', error);
      }
    } else {
      console.error('No file selected');
    }
  };

  return (
    <div className="App">
      <Header title="Closet Craft" subheader="Upload a picture to get a personal color recommendation!"/>
      <ImageUpload handleImageUpload={handleImageUpload} />
      {selectedImage && (
        <div className={styles.previewContainer}>
          <h2 className={styles.sectionTitle}>Preview:</h2>
          <img src={selectedImage} alt="Uploaded" className={styles.previewImage} />
        </div>
      )}
      {outfitRecommendation.category && (
        <ColorRecommendation
          category={outfitRecommendation.category}
          primary_color={outfitRecommendation.primary_color}
          color={outfitRecommendation.color}
        />
      )}
    </div>
  );
}

export default App;
