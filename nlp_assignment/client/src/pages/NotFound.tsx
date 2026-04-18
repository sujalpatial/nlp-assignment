import { useLocation } from 'wouter'

export default function NotFound() {
  const [, navigate] = useLocation()
  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <div className="text-center space-y-4">
        <h1 className="text-6xl font-bold text-foreground">404</h1>
        <p className="text-muted-foreground">Page not found</p>
        <button onClick={() => navigate('/')}
          className="px-4 py-2 bg-accent text-white rounded-lg hover:bg-accent/90">
          Go Home
        </button>
      </div>
    </div>
  )
}
