import React from 'react';
import styles from './ImageUpload.module.css';

const ImageUpload = ({ handleImageUpload }) => {
  return (
    <div className={styles.imageUpload}>
      <label htmlFor="fileInput">Upload Image</label>
      <input id="fileInput" type="file" accept="image/*" onChange={handleImageUpload} />
    </div>
  );
};

export default ImageUpload;
