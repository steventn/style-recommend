import React, { useState } from 'react';
import Header from './components/Header/Header.js';
import ImageUpload from './components/ImageUpload/ImageUpload.js';
import API_BASE_URL from "./config";
import axios from 'axios';


const App = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [outfitRecommendation, setOutfitRecommendation] = useState({ category: '', primary_color: '', colors: {} });


  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    setSelectedImage(URL.createObjectURL(file));

    if (file) {
      const formData = new FormData();
      formData.append('file', file); 
  
      try {
        const response = await axios.post(`${API_BASE_URL}/get_recommendation`, formData);
        console.log(response)

        if (response.data && response.data.category) {
          console.log(response.data.color)
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
      <Header title="Closet Craft" subheader="Upload a picture to get a personal style recommendation!"/>
      <ImageUpload handleImageUpload={handleImageUpload} />
      {selectedImage && (
        <div>
          <h2>Preview:</h2>
          <img src={selectedImage} alt="Uploaded" style={{ maxWidth: '100%' }} />
        </div>
      )}
      {outfitRecommendation.category && (
        <div>
          <h2>Outfit Recommendation:</h2>
          <p>Category: {outfitRecommendation.category}</p>
          <p>Primary Color: {outfitRecommendation.primary_color}</p>
          <p>Colors:</p>
          <ul>
            {outfitRecommendation.color.complementary_color && (
              <li>
                <h3>Complementary</h3>
                <ul>
                  {outfitRecommendation.color.complementary_color.map((color, index) => (
                    <li key={index}>
                      <div style={{ backgroundColor: `rgb(${color.join(',')})`, width: '50px', height: '50px' }}></div>
                      <p>{`(${color.join(', ')})`}</p>
                    </li>
                  ))}
                </ul>
              </li>
            )}
            {outfitRecommendation.color.monochromatic_colors && (
              <li>
                <h3>Monochromatic</h3>
                <ul>
                  {outfitRecommendation.color.monochromatic_colors.map((color, index) => (
                    <li key={index}>
                      <div style={{ backgroundColor: `rgb(${color.join(',')})`, width: '50px', height: '50px' }}></div>
                      <p>{`(${color.join(', ')})`}</p>
                    </li>
                  ))}
                </ul>
              </li>
            )}
            {outfitRecommendation.color.analogous_colors && (
              <li>
                <h3>Analogous</h3>
                <ul>
                  {outfitRecommendation.color.analogous_colors.map((color, index) => (
                    <li key={index}>
                      <div style={{ backgroundColor: `rgb(${color.join(',')})`, width: '50px', height: '50px' }}></div>
                      <p>{`(${color.join(', ')})`}</p>
                    </li>
                  ))}
                </ul>
              </li>
            )}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
