<template>
  <div id="app">
    <input type="file" @change="handleFileUpload" accept="image/*">
    <div v-if="colorRecommendation">
      <p>Color: {{ colorRecommendation.color }}</p>
      <p>Complementary Color: {{ colorRecommendation.complementary_color }}</p>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  data() {
    return {
      colorRecommendation: null,
    };
  },
  methods: {
    async handleFileUpload(event) {
      const file = event.target.files[0];

      // Convert the file to base64
      const reader = new FileReader();
      reader.onload = async () => {
        const base64Image = reader.result.split(',')[1];

        // Send the base64 image to the backend
        const response = await axios.post('http://localhost:5000/get_color_recommendation', {
          image_path: base64Image,
        });

        this.colorRecommendation = response.data;
      };

      reader.readAsDataURL(file);
    },
  },
};
</script>

<style>
#app {
  text-align: center;
  margin-top: 60px;
}
</style>
