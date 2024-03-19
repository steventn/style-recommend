import React from 'react';
import styles from './Header.module.css';

const Header = ({ title, subheader }) => {
  return (
    <div className={styles.header}>
      <h1 className={styles.title}>{title}</h1>
      <h2 className={styles.subheader}>{subheader}</h2>
    </div>
  );
};

export default Header;
