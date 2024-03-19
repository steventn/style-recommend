import React, { useState } from 'react';
import Header from './components/Header/Header.js';
import ImageUpload from './components/ImageUpload/ImageUpload.js';

const App = () => {
  const [selectedImage, setSelectedImage] = useState(null);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    setSelectedImage(URL.createObjectURL(file));
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
    </div>
  );
}

export default App;
