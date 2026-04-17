import { COOKIE_NAME } from "@shared/const";
import { getSessionCookieOptions } from "./_core/cookies";
import { systemRouter } from "./_core/systemRouter";
import { publicProcedure, router } from "./_core/trpc";
import { z } from "zod";

export const appRouter = router({
    // if you need to use socket.io, read and register route in server/_core/index.ts, all api should start with '/api/' so that the gateway can route correctly
  system: systemRouter,
  auth: router({
    me: publicProcedure.query(opts => opts.ctx.user),
    logout: publicProcedure.mutation(({ ctx }) => {
      const cookieOptions = getSessionCookieOptions(ctx.req);
      ctx.res.clearCookie(COOKIE_NAME, { ...cookieOptions, maxAge: -1 });
      return {
        success: true,
      } as const;
    }),
  }),

  inference: router({
    generateWithAnalysis: publicProcedure
      .input(z.object({
        prompt: z.string(),
        maxNewTokens: z.number().default(50),
        temperature: z.number().default(0.7),
      }))
      .mutation(async ({ input }) => {
        // This will be connected to the Python inference engine via WebSocket
        return {
          generatedText: "[Generated text will stream via WebSocket]",
          analyses: [],
        };
      }),
  }),
});

export type AppRouter = typeof appRouter;
