import {
    withStreamlitConnection,
    ComponentProps,
    Streamlit,
} from "streamlit-component-lib"
import {
    useEffect,
    ReactElement,
    useRef,
    useState
} from "react"
import WaveSurfer from "wavesurfer.js"

function base64ToBlob(base64String: string, contentType = '') {
    const byteCharacters = atob(base64String);
    const byteArrays = [];

    for (let i = 0; i < byteCharacters.length; i++) {
        byteArrays.push(byteCharacters.charCodeAt(i));
    }

    const byteArray = new Uint8Array(byteArrays);
    return new Blob([byteArray], { type: contentType });
}

function MyComponent({ args, disabled, theme }: ComponentProps): ReactElement {
    // Extract custom arguments passed from Python
    const { audio } = args

    // Component state
    const waveformRef = useRef<HTMLDivElement>(null);
    const wavesurferRef = useRef<WaveSurfer | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    function _initializeWaveSurfer() {
        if (waveformRef.current === null) {
            return;
        }
        _cleanCurrentWavesurfer();
        const wavesurfer = WaveSurfer.create({
            container: waveformRef.current,
            barWidth: 3,
            barRadius: 3,
            barGap: 2,
            barHeight: 1,
            cursorWidth: 1,
            backend: "WebAudio",
            height: 80,
            progressColor: "#FE6E00",
            waveColor: "#C4C4C4",
            cursorColor: "transparent"
        });

        wavesurferRef.current = wavesurfer;
    }

    function _cleanCurrentWavesurfer() {
        if (!wavesurferRef.current) {
            return;
        }
        wavesurferRef.current.destroy();
        wavesurferRef.current = null;
    }

    function _loadAudio() {
        if (wavesurferRef.current === null) {
            console.error("Trying to load audio with no wavesurfer instance");
            return;
        }
        const audio_url = URL.createObjectURL(base64ToBlob(audio));
        wavesurferRef.current.load(audio_url);
    }

    useEffect(() => {
        _cleanCurrentWavesurfer();

        setIsLoading(true);
        setError(null);

        setTimeout(() => {
            try {
                console.log("Initializing wavesurfer...")
                _initializeWaveSurfer();

                _loadAudio();

                Streamlit.setFrameHeight(waveformRef.current?.clientHeight);
                setIsLoading(false);
            } catch (error) {
                console.error("Error initializing WaveSurfer:", error);
                setError("Failed to initialize audio player: " + (error as Error).message);
                setIsLoading(false);
            }
        }, 100);

        return _cleanCurrentWavesurfer();
    }, [audio])

    return (
        <div ref={waveformRef} style={{
            padding: '20px',
        }}>
            <h1>Music player</h1>
            {isLoading && <p style={{ margin: '10px 0' }}>Loading audio...</p>}
            {error && <p style={{ color: 'red', margin: '10px 0' }}>Error: {error}</p>}

            <button
                onClick={() => {
                    console.log("Play button clicked");
                    wavesurferRef.current?.playPause();
                }}
                style={{
                    padding: '8px 16px',
                    backgroundColor: '#FE6E00',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontSize: '14px'
                }}
            >
                Play/Pause
            </button>
        </div>
    )
}

export default withStreamlitConnection(MyComponent)
