import Head from 'next/head';
import { useState } from 'react';
import styles from '../styles/Home.module.css';

export default function Home() {
  const [prompt, setPrompt] = useState('');
  const [images, setImages] = useState([]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!prompt) return;

    try {
      const res = await fetch('http://localhost:8000/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      });
      const data = await res.json();
      if (data.success && data.images) {
        setImages(data.images);
      } else {
        alert(data.error || 'Failed to generate image');
      }
    } catch (err) {
      /* eslint-disable no-console */
      console.error(err);
    }
  };

  return (
    <div className={styles.container}>
      <Head>
        <title>AI Image Generator</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className={styles.main}>
        <h1 className={styles.title}>AI Image Generator</h1>

        <form onSubmit={handleSubmit} className={styles.form}>
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter prompt"
          />
          <button type="submit">Generate</button>
        </form>

        <div className={styles.results}>
          {images.map((img, idx) => (
            <img
              key={idx}
              src={`data:image/png;base64,${img}`}
              alt={`generated-${idx}`}
            />
          ))}
        </div>
      </main>
    </div>
  );
}
