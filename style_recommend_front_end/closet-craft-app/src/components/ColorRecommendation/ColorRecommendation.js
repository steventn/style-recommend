import React from 'react';
import styles from './ColorRecommendation.module.css';

const ColorRecommendations = ({ category, primary_color, color }) => {
  return (
    <div className={styles.container}>
      <h2 className={styles.categoryHeading}>Identification:</h2>
      <p>Category: {category}</p>
      <p>Primary Color: {primary_color}</p>
      <h3 className={styles.suggestionHeading}>Suggested Colors To Pair: </h3>
      <ul className={styles.colorList}>
        {color.complementary_color && (
          <li className={styles.colorCategory}>
            <h3 className={styles.categoryHeading}>Complementary</h3>
            <ul>
              {color.complementary_color.map((color, index) => (
                <li key={index} className={styles.colorItem}>
                  <div className={styles.colorBox} style={{ backgroundColor: `rgb(${color.join(',')})` }}></div>
                  <p className={styles.colorText}>{`(${color.join(', ')})`}</p>
                </li>
              ))}
            </ul>
          </li>
        )}
        {color.monochromatic_colors && (
          <li className={styles.colorCategory}>
            <h3 className={styles.categoryHeading}>Monochromatic</h3>
            <ul>
              {color.monochromatic_colors.map((color, index) => (
                <li key={index} className={styles.colorItem}>
                  <div className={styles.colorBox} style={{ backgroundColor: `rgb(${color.join(',')})` }}></div>
                  <p className={styles.colorText}>{`(${color.join(', ')})`}</p>
                </li>
              ))}
            </ul>
          </li>
        )}
        {color.analogous_colors && (
          <li className={styles.colorCategory}>
            <h3 className={styles.categoryHeading}>Analogous</h3>
            <ul>
              {color.analogous_colors.map((color, index) => (
                <li key={index} className={styles.colorItem}>
                  <div className={styles.colorBox} style={{ backgroundColor: `rgb(${color.join(',')})` }}></div>
                  <p className={styles.colorText}>{`(${color.join(', ')})`}</p>
                </li>
              ))}
            </ul>
          </li>
        )}
      </ul>
    </div>
  );
};

export default ColorRecommendations;
