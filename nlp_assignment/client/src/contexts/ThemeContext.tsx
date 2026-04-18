import React, { createContext, useContext, useState } from 'react'

type Theme = 'light' | 'dark'
interface ThemeContextValue { theme: Theme; setTheme: (t: Theme) => void }

const ThemeContext = createContext<ThemeContextValue>({ theme: 'light', setTheme: () => {} })

export function useTheme() { return useContext(ThemeContext) }

export function ThemeProvider({ children, defaultTheme = 'light' }: {
  children: React.ReactNode; defaultTheme?: Theme; switchable?: boolean
}) {
  const [theme, setTheme] = useState<Theme>(defaultTheme)
  React.useEffect(() => {
    document.documentElement.classList.remove('light', 'dark')
    document.documentElement.classList.add(theme)
  }, [theme])
  return <ThemeContext.Provider value={{ theme, setTheme }}>{children}</ThemeContext.Provider>
}
